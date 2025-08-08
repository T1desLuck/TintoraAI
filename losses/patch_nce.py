"""
PatchNCE Loss: Контрастное + градиентное обучение для колоризации.

Данный модуль реализует PatchNCE (Patch-based Noise Contrastive Estimation) - 
контрастную функцию потерь, которая помогает модели различать правильные 
и неправильные соответствия между цветовыми пространствами.

Ключевые особенности:
- Контрастное обучение на уровне патчей для сохранения локальных цветовых паттернов
- Градиентное обучение для лучшего сохранения непрерывности цветовых переходов
- Многоуровневая структура для работы с признаками разных масштабов
- Адаптивная температура для настройки "жесткости" контраста

Преимущества для колоризации:
- Улучшает локальную согласованность цветов
- Помогает избежать "размытых" или "усредненных" цветовых решений
- Способствует более реалистичным и насыщенным цветам
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PatchNCELoss(nn.Module):
    """
    Базовая PatchNCE функция потерь для контрастного + градиентного обучения.
    
    Реализует простое PatchNCE контрастное обучение с градиентной составляющей
    для сохранения локальных цветовых паттернов и непрерывности переходов.
    
    Args:
        temperature (float): Параметр температуры для контрастного обучения
        patch_size (int): Размер патчей для выборки
        n_patches (int): Количество патчей для выборки из каждого изображения
        device (torch.device): Устройство для вычислений
    """
    def __init__(self, temperature=0.07, patch_size=16, n_patches=256, device=None):
        super().__init__()
        self.temperature = temperature
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.device = device if device is not None else torch.device('cpu')
        
        # Создаем сэмплер патчей
        self.patch_sampler = PatchSampler(
            patch_size=patch_size,
            patch_count=n_patches,
            use_perceptual_sampling=True
        )
        
        # Проектор для преобразования патчей в контрастное пространство
        # Адаптивно определяем размерность входных признаков
        self.projector = None  # Будет инициализирован при первом вызове
        self.projection_dim = 128  # Размерность проекционного пространства
        
        # Критерий потерь
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def _init_projector(self, feature_dim):
        """
        Инициализирует проектор на основе размерности входных признаков.
        
        Args:
            feature_dim (int): Размерность входных признаков
        """
        if self.projector is None:
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim)
            ).to(self.device)
    
    def extract_patches(self, img):
        """
        Извлекает патчи из изображения.
        
        Args:
            img (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Извлеченные патчи [B, n_patches, patch_size*patch_size*C]
        """
        batch_size, channels, height, width = img.shape
        
        # Используем сэмплер для извлечения патчей
        patches = self.patch_sampler(img)  # [B, n_patches, C, patch_size, patch_size]
        
        # Преобразуем в плоский формат
        patches_flat = patches.reshape(
            batch_size, 
            self.n_patches, 
            channels * self.patch_size * self.patch_size
        )
        
        return patches_flat
    
    def _compute_contrastive_loss(self, anchor_features, positive_features, negative_features):
        """
        Вычисляет контрастную потерю между якорными, положительными и отрицательными признаками.
        
        Args:
            anchor_features (torch.Tensor): Якорные признаки [B, N, D]
            positive_features (torch.Tensor): Положительные признаки [B, N, D]
            negative_features (torch.Tensor): Отрицательные признаки [B, N_neg, D]
            
        Returns:
            torch.Tensor: Контрастная потеря
        """
        batch_size, n_patches, feature_dim = anchor_features.shape
        total_loss = 0.0
        
        for b in range(batch_size):
            anchor = anchor_features[b]  # [N, D]
            positive = positive_features[b]  # [N, D]
            
            # Нормализуем признаки
            anchor = F.normalize(anchor, dim=1)
            positive = F.normalize(positive, dim=1)
            
            # Вычисляем сходство между якорными и положительными патчами
            pos_similarity = torch.sum(anchor * positive, dim=1) / self.temperature  # [N]
            
            # Вычисляем сходство с отрицательными примерами
            if negative_features is not None:
                negative = F.normalize(negative_features[b], dim=1)  # [N_neg, D]
                neg_similarity = torch.matmul(anchor, negative.transpose(0, 1)) / self.temperature  # [N, N_neg]
                
                # Объединяем положительные и отрицательные сходства
                logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)  # [N, 1+N_neg]
                labels = torch.zeros(n_patches, dtype=torch.long, device=self.device)  # Положительные на позиции 0
            else:
                # Если нет отрицательных примеров, используем только положительные
                logits = pos_similarity.unsqueeze(1)  # [N, 1]
                labels = torch.zeros(n_patches, dtype=torch.long, device=self.device)
            
            # Вычисляем cross-entropy потерю
            loss = self.cross_entropy(logits, labels)
            total_loss += loss
        
        return total_loss / batch_size
    
    def _compute_gradient_loss(self, query, reference):
        """
        Вычисляет градиентную потерю для сохранения непрерывности цветовых переходов.
        
        Args:
            query (torch.Tensor): Запросное изображение [B, C, H, W]
            reference (torch.Tensor): Эталонное изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Градиентная потеря
        """
        def compute_gradients(x):
            # Горизонтальные градиенты
            h_grad = x[:, :, :, 1:] - x[:, :, :, :-1]
            # Вертикальные градиенты
            v_grad = x[:, :, 1:, :] - x[:, :, :-1, :]
            return h_grad, v_grad
        
        # Вычисляем градиенты для запросного и эталонного изображений
        query_h_grad, query_v_grad = compute_gradients(query)
        ref_h_grad, ref_v_grad = compute_gradients(reference)
        
        # Вычисляем L1 потерю между градиентами
        h_loss = F.l1_loss(query_h_grad, ref_h_grad)
        v_loss = F.l1_loss(query_v_grad, ref_v_grad)
        
        return h_loss + v_loss
    
    def forward(self, query, key, reference):
        """
        Прямое распространение для вычисления PatchNCE потери.
        
        Args:
            query (torch.Tensor): Запросное изображение (обычно L-канал) [B, C_q, H, W]
            key (torch.Tensor): Ключевое изображение (обычно предсказанные ab-каналы) [B, C_k, H, W] 
            reference (torch.Tensor): Эталонное изображение (целевые ab-каналы) [B, C_r, H, W]
            
        Returns:
            torch.Tensor: Скалярная потеря
        """
        # Перемещаем на нужное устройство
        query = query.to(self.device)
        key = key.to(self.device)
        reference = reference.to(self.device)
        
        # Извлекаем патчи из всех изображений
        query_patches = self.extract_patches(query)  # [B, N, D_q]
        key_patches = self.extract_patches(key)      # [B, N, D_k]
        ref_patches = self.extract_patches(reference) # [B, N, D_r]
        
        # Определяем размерности признаков и инициализируем проекторы при необходимости
        query_feature_dim = query_patches.shape[-1]
        key_feature_dim = key_patches.shape[-1]
        ref_feature_dim = ref_patches.shape[-1]
        
        # Для простоты используем проектор с максимальной размерностью
        max_feature_dim = max(query_feature_dim, key_feature_dim, ref_feature_dim)
        self._init_projector(max_feature_dim)
        
        # Приводим все патчи к одинаковой размерности через padding или обрезку
        def pad_or_crop_to_size(patches, target_dim):
            current_dim = patches.shape[-1]
            if current_dim < target_dim:
                # Дополняем нулями
                padding = torch.zeros(
                    patches.shape[0], patches.shape[1], target_dim - current_dim,
                    device=patches.device, dtype=patches.dtype
                )
                return torch.cat([patches, padding], dim=-1)
            elif current_dim > target_dim:
                # Обрезаем
                return patches[:, :, :target_dim]
            else:
                return patches
        
        query_patches = pad_or_crop_to_size(query_patches, max_feature_dim)
        key_patches = pad_or_crop_to_size(key_patches, max_feature_dim)
        ref_patches = pad_or_crop_to_size(ref_patches, max_feature_dim)
        
        # Проецируем патчи в контрастное пространство
        batch_size, n_patches = query_patches.shape[:2]
        
        query_projected = self.projector(query_patches.reshape(-1, max_feature_dim))
        query_projected = query_projected.reshape(batch_size, n_patches, self.projection_dim)
        
        key_projected = self.projector(key_patches.reshape(-1, max_feature_dim))
        key_projected = key_projected.reshape(batch_size, n_patches, self.projection_dim)
        
        ref_projected = self.projector(ref_patches.reshape(-1, max_feature_dim))
        ref_projected = ref_projected.reshape(batch_size, n_patches, self.projection_dim)
        
        # Вычисляем контрастную потерю
        # Якорь: запрос, положительные: ключи, отрицательные: эталон из других изображений
        contrastive_loss = self._compute_contrastive_loss(
            anchor_features=query_projected,
            positive_features=key_projected,
            negative_features=ref_projected
        )
        
        # Вычисляем градиентную потерю между ключом и эталоном
        gradient_loss = self._compute_gradient_loss(key, reference)
        
        # Общая потеря: контрастная + градиентная (с весовым коэффициентом)
        total_loss = contrastive_loss + 0.5 * gradient_loss
        
        return total_loss


class PatchSampler(nn.Module):
    """
    Модуль для выбора патчей из карт признаков.
    
    Args:
        patch_size (int): Размер выбираемых патчей
        patch_count (int): Количество патчей для выбора из карты признаков
        use_perceptual_sampling (bool): Использовать ли перцептивную выборку (большая вероятность для
                                       патчей с высоким градиентом или сложной текстурой)
    """
    def __init__(self, patch_size=1, patch_count=256, use_perceptual_sampling=True):
        super().__init__()
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.use_perceptual_sampling = use_perceptual_sampling
        
    def forward(self, feat, attention_map=None):
        """
        Выбирает патчи из карты признаков.
        
        Args:
            feat (torch.Tensor): Тензор признаков [B, C, H, W]
            attention_map (torch.Tensor, optional): Карта внимания для взвешенного выбора [B, 1, H, W]
            
        Returns:
            torch.Tensor: Выбранные патчи [B, patch_count, C, patch_size, patch_size]
        """
        B, C, H, W = feat.shape
        
        # Проверяем, что размер признаков достаточно большой для выбора патчей
        assert H >= self.patch_size and W >= self.patch_size, \
            f"Размер признаков ({H}x{W}) должен быть больше или равен размеру патчей ({self.patch_size})"
        
        if self.patch_size > 1:
            # Если размер патча больше 1, используем unfold для получения перекрывающихся патчей
            # Примечание: stride=1 означает полное перекрытие
            patches = F.unfold(feat, kernel_size=self.patch_size, stride=1)
            # Преобразуем в [B, C*patch_size*patch_size, H*W]
            patches = patches.view(B, C, self.patch_size*self.patch_size, -1)
            # Перестановка для получения [B, H*W, C, patch_size*patch_size]
            patches = patches.permute(0, 3, 1, 2)
            # Изменяем форму для получения [B, H*W, C, patch_size, patch_size]
            patches = patches.view(B, -1, C, self.patch_size, self.patch_size)
        else:
            # Если размер патча 1, просто изменяем форму тензора
            patches = feat.permute(0, 2, 3, 1).reshape(B, -1, C).unsqueeze(-1).unsqueeze(-1)
        
        num_available_patches = patches.shape[1]
        
        if self.use_perceptual_sampling and attention_map is not None:
            # Перцептивная выборка на основе карты внимания
            # Преобразуем карту внимания в веса для выборки
            weights = F.interpolate(attention_map, size=(H, W), mode='bilinear', align_corners=False)
            weights = weights.view(B, -1)
            
            # Нормализуем веса для получения распределения вероятностей
            weights = F.softmax(weights, dim=1)
            
            # Случайная выборка с учетом весов
            selected_indices = torch.multinomial(weights, 
                                               min(self.patch_count, num_available_patches),
                                               replacement=False)
        else:
            # Равномерная случайная выборка без замены
            selected_indices = torch.randperm(num_available_patches, device=feat.device)[:min(self.patch_count, num_available_patches)]
            # Повторяем индексы для каждого элемента в батче
            selected_indices = selected_indices.unsqueeze(0).expand(B, -1)
        
        # Выбираем патчи по индексам
        selected_patches = torch.gather(patches, 1, 
                                       selected_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                                           -1, -1, C, self.patch_size, self.patch_size))
        
        return selected_patches


class MultiScalePatchNCELoss(nn.Module):
    """
    Многомасштабная PatchNCE функция потерь для контрастного обучения на разных уровнях.
    
    Args:
        feature_dims (list): Список размерностей признаков для каждого уровня
        patch_sizes (list): Список размеров патчей для каждого уровня
        patch_counts (list): Список количеств патчей для каждого уровня
        temperature (float): Параметр температуры для контрастного обучения
        negative_samples_strategy (str): Стратегия выбора негативных примеров ('batch', 'memory', 'mixed')
        memory_bank_size (int): Размер банка памяти для стратегии 'memory'
        use_positives_as_negatives (bool): Использовать ли положительные примеры из других изображений как негативные
        adaptive_temperature (bool): Использовать ли адаптивную температуру
    """
    def __init__(
        self,
        feature_dims=[256, 512, 1024],
        patch_sizes=[1, 3, 5],
        patch_counts=[256, 128, 64],
        temperature=0.07,
        negative_samples_strategy='batch',  # 'batch', 'memory', 'mixed'
        memory_bank_size=4096,
        use_positives_as_negatives=True,
        adaptive_temperature=True
    ):
        super().__init__()
        
        self.num_layers = len(feature_dims)
        assert self.num_layers == len(patch_sizes) and self.num_layers == len(patch_counts), \
            "Размеры списков feature_dims, patch_sizes и patch_counts должны совпадать"
        
        # Базовая температура
        self.base_temperature = temperature
        self.temperature = nn.Parameter(torch.ones(self.num_layers) * temperature)
        self.adaptive_temperature = adaptive_temperature
        
        # Проекторы для каждого уровня признаков (линейный проектор с нормализацией)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 256),  # Проецируем все уровни в общее пространство размерности 256
                nn.BatchNorm1d(256)
            ) for dim in feature_dims
        ])
        
        # Выборщики патчей для каждого уровня
        self.patch_samplers = nn.ModuleList([
            PatchSampler(
                patch_size=patch_sizes[i],
                patch_count=patch_counts[i],
                use_perceptual_sampling=(i < self.num_layers // 2)  # Перцептивная выборка для нижних уровней
            ) for i in range(self.num_layers)
        ])
        
        # Параметры для стратегии негативных примеров
        self.negative_samples_strategy = negative_samples_strategy
        self.use_positives_as_negatives = use_positives_as_negatives
        
        # Инициализация банка памяти при необходимости
        if negative_samples_strategy in ['memory', 'mixed']:
            self.register_buffer('memory_bank', torch.randn(memory_bank_size, 256))
            self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
            # Нормализуем банк памяти
            with torch.no_grad():
                self.memory_bank = F.normalize(self.memory_bank, dim=1)
                
        # Веса для каждого уровня признаков
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
        # Критерий для расчета потерь
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Счетчик для отслеживания сложных примеров
        self.register_buffer('hard_negative_count', torch.zeros(1))
        self.register_buffer('total_count', torch.zeros(1))
        
    def _get_anchor_positive_pairs(self, anchor_feats, positive_feats, attention_maps=None):
        """
        Получает пары патчей "якорь-положительный пример" из карт признаков.
        
        Args:
            anchor_feats (list): Список тензоров признаков-якорей для каждого уровня
            positive_feats (list): Список тензоров признаков положительных примеров для каждого уровня
            attention_maps (list, optional): Список карт внимания для каждого уровня
            
        Returns:
            list: Список кортежей (anchor_patches, positive_patches) для каждого уровня
        """
        pairs = []
        
        for i in range(self.num_layers):
            # Получаем патчи из якорных и положительных признаков
            anchor_patches = self.patch_samplers[i](anchor_feats[i], 
                                                 attention_maps[i] if attention_maps is not None else None)
            positive_patches = self.patch_samplers[i](positive_feats[i], 
                                                   attention_maps[i] if attention_maps is not None else None)
            
            pairs.append((anchor_patches, positive_patches))
            
        return pairs
    
    def _get_negative_samples(self, all_patches, anchor_idx, positive_idx):
        """
        Получает негативные примеры в зависимости от выбранной стратегии.
        
        Args:
            all_patches (torch.Tensor): Тензор всех патчей [B, N, D]
            anchor_idx (int): Индекс якорного примера
            positive_idx (int): Индекс положительного примера
            
        Returns:
            torch.Tensor: Тензор негативных примеров [N_neg, D]
        """
        batch_size = all_patches.shape[0]
        
        if self.negative_samples_strategy == 'batch':
            # Используем все патчи из текущего батча как негативные
            negatives = []
            
            # Добавляем патчи из других изображений
            for i in range(batch_size):
                if i != anchor_idx:  # Исключаем якорное изображение
                    negatives.append(all_patches[i])
                    
            # Если разрешено, добавляем патчи из положительного изображения
            if self.use_positives_as_negatives and positive_idx != anchor_idx:
                negatives.append(all_patches[positive_idx])
                
            if negatives:
                negatives = torch.cat(negatives, dim=0)
            else:
                # Если нет негативных примеров (маленький батч), создаем случайные
                negatives = torch.randn_like(all_patches[0])
                
        elif self.negative_samples_strategy == 'memory':
            # Используем банк памяти как негативные примеры
            negatives = self.memory_bank
            
        else:  # 'mixed'
            # Используем как банк памяти, так и примеры из батча
            batch_negatives = []
            
            for i in range(batch_size):
                if i != anchor_idx:
                    batch_negatives.append(all_patches[i])
                    
            if self.use_positives_as_negatives and positive_idx != anchor_idx:
                batch_negatives.append(all_patches[positive_idx])
                
            if batch_negatives:
                batch_negatives = torch.cat(batch_negatives, dim=0)
                # Объединяем с примерами из банка памяти
                negatives = torch.cat([batch_negatives, self.memory_bank], dim=0)
            else:
                negatives = self.memory_bank
                
        return negatives
    
    def _update_memory_bank(self, features):
        """
        Обновляет банк памяти новыми признаками.
        
        Args:
            features (torch.Tensor): Тензор новых признаков [B*N, D]
        """
        if self.negative_samples_strategy not in ['memory', 'mixed']:
            return
            
        batch_size = features.shape[0]
        memory_size = self.memory_bank.shape[0]
        
        # Получаем текущую позицию в банке памяти
        ptr = int(self.memory_ptr)
        
        # Обновляем банк памяти
        if ptr + batch_size <= memory_size:
            # Простое обновление
            self.memory_bank[ptr:ptr+batch_size] = features
            ptr += batch_size
        else:
            # Циклическое обновление
            remaining = memory_size - ptr
            self.memory_bank[ptr:] = features[:remaining]
            self.memory_bank[:batch_size-remaining] = features[remaining:]
            ptr = (ptr + batch_size) % memory_size
            
        # Обновляем указатель
        self.memory_ptr[0] = ptr
    
    def _compute_patch_nce_loss(self, anchor_patches, positive_patches, layer_idx):
        """
        Вычисляет PatchNCE loss для одного уровня признаков.
        
        Args:
            anchor_patches (torch.Tensor): Патчи-якоря [B, N, C, patch_size, patch_size]
            positive_patches (torch.Tensor): Позитивные патчи [B, N, C, patch_size, patch_size]
            layer_idx (int): Индекс слоя для выбора правильной температуры
            
        Returns:
            torch.Tensor: Значение функции потерь
            float: Доля сложных негативных примеров
        """
        batch_size, num_patches = anchor_patches.shape[0], anchor_patches.shape[1]
        feature_dim = anchor_patches.shape[2]
        patch_size = anchor_patches.shape[3]
        
        # Сглаживаем размерность патчей
        anchor_patches = anchor_patches.reshape(batch_size, num_patches, feature_dim * patch_size * patch_size)
        positive_patches = positive_patches.reshape(batch_size, num_patches, feature_dim * patch_size * patch_size)
        
        # Проходим через проектор
        anchor_features = []
        positive_features = []
        
        for i in range(batch_size):
            # Проецируем патчи
            anchor_proj = self.projectors[layer_idx](anchor_patches[i])  # [N, 256]
            positive_proj = self.projectors[layer_idx](positive_patches[i])  # [N, 256]
            
            # Нормализуем признаки
            anchor_proj = F.normalize(anchor_proj, dim=1)
            positive_proj = F.normalize(positive_proj, dim=1)
            
            anchor_features.append(anchor_proj)
            positive_features.append(positive_proj)
        
        # Инициализируем общую потерю
        total_loss = 0.0
        hard_negatives = 0
        total_samples = 0
        
        # Вычисляем потери для каждого изображения в батче
        for anchor_idx in range(batch_size):
            anchor_feat = anchor_features[anchor_idx]  # [N, 256]
            
            for positive_idx in range(batch_size):
                # Если мы используем один и тот же индекс, это соответствующие патчи
                # Иначе это разные изображения (также могут использоваться как положительные примеры)
                if anchor_idx == positive_idx:
                    positive_feat = positive_features[positive_idx]  # [N, 256]
                    
                    # Вычисляем лог-правдоподобие для каждой пары патчей
                    logits = torch.matmul(anchor_feat, positive_feat.transpose(0, 1))  # [N, N]
                    
                    # Температурное масштабирование
                    temperature = self.temperature[layer_idx] if self.adaptive_temperature else self.base_temperature
                    logits = logits / temperature
                    
                    # Метки - диагональ матрицы (соответствующие патчи имеют одинаковые индексы)
                    labels = torch.arange(num_patches, device=logits.device)
                    
                    # Добавляем негативные примеры
                    if self.negative_samples_strategy != 'batch' or batch_size > 1:
                        # Получаем все патчи для использования в качестве негативных примеров
                        all_patches = torch.cat([f for f in anchor_features], dim=0)  # [B*N, 256]
                        
                        # Получаем негативные примеры
                        negatives = self._get_negative_samples(all_patches, anchor_idx, positive_idx)  # [N_neg, 256]
                        
                        # Вычисляем сходство с негативными примерами
                        negative_logits = torch.matmul(anchor_feat, negatives.transpose(0, 1))  # [N, N_neg]
                        negative_logits = negative_logits / temperature
                        
                        # Объединяем логиты положительных и отрицательных примеров
                        combined_logits = torch.cat([logits, negative_logits], dim=1)  # [N, N+N_neg]
                        
                        # Обновляем метки (положительные примеры остаются на диагонали)
                        labels = torch.arange(num_patches, device=combined_logits.device)
                        
                        # Вычисляем потерю
                        loss = self.criterion(combined_logits, labels)
                        
                        # Анализируем сложные негативные примеры (те, которые имеют более высокий скор, чем позитивные)
                        with torch.no_grad():
                            for i in range(num_patches):
                                pos_score = logits[i, i]  # Скор позитивного примера
                                neg_scores = negative_logits[i]  # Скоры негативных примеров
                                hard_neg_count = torch.sum(neg_scores > pos_score).item()
                                hard_negatives += hard_neg_count
                                total_samples += neg_scores.shape[0]
                    else:
                        # Если у нас только одно изображение в батче и стратегия 'batch', просто используем контрастную потерю
                        loss = self.criterion(logits, labels)
                    
                    # Добавляем потерю для текущей пары изображений
                    total_loss += loss.mean()
        
        # Нормализуем потерю по количеству пар изображений
        normalized_loss = total_loss / (batch_size * batch_size)
        
        # Вычисляем долю сложных негативных примеров
        hard_negative_ratio = hard_negatives / max(total_samples, 1)
        
        # Обновляем счетчики сложных примеров
        self.hard_negative_count += hard_negatives
        self.total_count += total_samples
        
        return normalized_loss, hard_negative_ratio
    
    def _compute_gradient_loss(self, anchor_feats, positive_feats, layer_idx):
        """
        Вычисляет потери на градиентах для сохранения непрерывности цветовых переходов.
        
        Args:
            anchor_feats (torch.Tensor): Якорные признаки [B, C, H, W]
            positive_feats (torch.Tensor): Позитивные признаки [B, C, H, W]
            layer_idx (int): Индекс слоя
            
        Returns:
            torch.Tensor: Значение функции потерь на градиентах
        """
        # Вычисляем градиенты по вертикали и горизонтали
        def compute_gradients(x):
            # Горизонтальный градиент (разница между соседними пикселями по ширине)
            h_gradient = x[:, :, :, 1:] - x[:, :, :, :-1]
            # Вертикальный градиент (разница между соседними пикселями по высоте)
            v_gradient = x[:, :, 1:, :] - x[:, :, :-1, :]
            return h_gradient, v_gradient
        
        # Вычисляем градиенты для якорных и позитивных признаков
        anchor_h_grad, anchor_v_grad = compute_gradients(anchor_feats)
        positive_h_grad, positive_v_grad = compute_gradients(positive_feats)
        
        # Вычисляем L1 потерю между градиентами
        h_loss = F.l1_loss(anchor_h_grad, positive_h_grad)
        v_loss = F.l1_loss(anchor_v_grad, positive_v_grad)
        
        # Общая потеря на градиентах
        gradient_loss = h_loss + v_loss
        
        return gradient_loss
    
    def update_temperature(self, hard_negative_ratio):
        """
        Обновляет параметр температуры на основе доли сложных негативных примеров.
        
        Args:
            hard_negative_ratio (float): Доля сложных негативных примеров
        """
        if not self.adaptive_temperature:
            return
            
        with torch.no_grad():
            # Если доля сложных негативных примеров высока, уменьшаем температуру (увеличиваем контраст)
            # Если низка, увеличиваем температуру (уменьшаем контраст)
            target_ratio = 0.2  # Целевая доля сложных негативных примеров
            adjustment_factor = 0.01  # Коэффициент коррекции
            
            for i in range(self.num_layers):
                adjustment = adjustment_factor * (hard_negative_ratio - target_ratio)
                new_temp = self.temperature[i] - adjustment
                # Ограничиваем температуру в разумных пределах
                self.temperature[i] = torch.clamp(new_temp, min=0.01, max=0.5)
    
    def get_hard_negative_stats(self):
        """
        Возвращает статистику сложных негативных примеров.
        
        Returns:
            float: Доля сложных негативных примеров с момента последнего сброса
        """
        if self.total_count > 0:
            ratio = (self.hard_negative_count / self.total_count).item()
        else:
            ratio = 0.0
            
        # Сбрасываем счетчики
        self.hard_negative_count.zero_()
        self.total_count.zero_()
        
        return ratio
    
    def forward(self, anchor_feats, positive_feats, attention_maps=None):
        """
        Прямое распространение для вычисления PatchNCE потерь.
        
        Args:
            anchor_feats (list): Список тензоров признаков-якорей для каждого уровня [B, C_i, H_i, W_i]
            positive_feats (list): Список тензоров признаков положительных примеров [B, C_i, H_i, W_i]
            attention_maps (list, optional): Список карт внимания для каждого уровня [B, 1, H_i, W_i]
            
        Returns:
            dict: {
                'total_loss': torch.Tensor,  # Общая потеря
                'nce_loss': torch.Tensor,    # Контрастная потеря
                'gradient_loss': torch.Tensor,  # Потеря на градиентах
                'layer_losses': list,  # Потери для каждого уровня
                'hard_negative_ratio': float  # Доля сложных негативных примеров
            }
        """
        assert len(anchor_feats) == self.num_layers, f"Ожидается {self.num_layers} уровней признаков"
        assert len(positive_feats) == self.num_layers, f"Ожидается {self.num_layers} уровней признаков"
        
        if attention_maps is not None:
            assert len(attention_maps) == self.num_layers, f"Ожидается {self.num_layers} карт внимания"
        
        # Получаем пары патчей для каждого уровня
        patch_pairs = self._get_anchor_positive_pairs(anchor_feats, positive_feats, attention_maps)
        
        # Вычисляем потери для каждого уровня
        layer_losses = []
        nce_losses = []
        gradient_losses = []
        hard_negative_ratios = []
        
        for i, (anchor_patches, positive_patches) in enumerate(patch_pairs):
            # Вычисляем PatchNCE потерю
            nce_loss, hard_negative_ratio = self._compute_patch_nce_loss(anchor_patches, positive_patches, i)
            nce_losses.append(nce_loss)
            hard_negative_ratios.append(hard_negative_ratio)
            
            # Вычисляем потерю на градиентах
            gradient_loss = self._compute_gradient_loss(anchor_feats[i], positive_feats[i], i)
            gradient_losses.append(gradient_loss)
            
            # Общая потеря для уровня (взвешенная сумма NCE и градиентной потери)
            # Балансируем контрастную потерю и потерю на градиентах
            layer_loss = nce_loss + 0.5 * gradient_loss
            layer_losses.append(layer_loss)
            
        # Вычисляем веса для каждого уровня (с помощью softmax для нормализации)
        normalized_weights = F.softmax(self.layer_weights, dim=0)
        
        # Вычисляем взвешенные потери
        total_nce_loss = sum(w * loss for w, loss in zip(normalized_weights, nce_losses))
        total_gradient_loss = sum(w * loss for w, loss in zip(normalized_weights, gradient_losses))
        total_loss = sum(w * loss for w, loss in zip(normalized_weights, layer_losses))
        
        # Вычисляем среднюю долю сложных негативных примеров
        avg_hard_negative_ratio = sum(hard_negative_ratios) / len(hard_negative_ratios)
        
        # Обновляем температуру, если необходимо
        self.update_temperature(avg_hard_negative_ratio)
        
        # Обновляем банк памяти, если используется
        if self.negative_samples_strategy in ['memory', 'mixed']:
            # Собираем все признаки для обновления банка памяти
            all_features = []
            
            for i in range(self.num_layers):
                anchor_patches = patch_pairs[i][0]
                batch_size, num_patches = anchor_patches.shape[0], anchor_patches.shape[1]
                feature_dim = anchor_patches.shape[2]
                patch_size = anchor_patches.shape[3]
                
                # Сглаживаем размерность патчей
                flattened = anchor_patches.reshape(batch_size, num_patches, feature_dim * patch_size * patch_size)
                
                # Выбираем случайное подмножество патчей для обновления банка памяти
                num_to_update = min(256 // self.num_layers, num_patches)
                
                for b in range(batch_size):
                    indices = torch.randperm(num_patches, device=flattened.device)[:num_to_update]
                    selected = flattened[b, indices]
                    
                    # Проецируем и нормализуем
                    projected = self.projectors[i](selected)
                    normalized = F.normalize(projected, dim=1)
                    
                    all_features.append(normalized)
            
            # Объединяем и обновляем банк памяти
            if all_features:
                all_features = torch.cat(all_features, dim=0)
                self._update_memory_bank(all_features)
        
        return {
            'total_loss': total_loss,
            'nce_loss': total_nce_loss,
            'gradient_loss': total_gradient_loss,
            'layer_losses': layer_losses,
            'hard_negative_ratio': avg_hard_negative_ratio
        }


class MultiScalePatchwiseLoss(nn.Module):
    """
    Комплексная функция потерь, объединяющая PatchNCE с другими полезными для колоризации критериями.
    
    Args:
        feature_dims (list): Список размерностей признаков для каждого уровня
        patch_sizes (list): Список размеров патчей для каждого уровня
        patch_counts (list): Список количеств патчей для каждого уровня
        temperature (float): Параметр температуры для контрастного обучения
        use_gradient_loss (bool): Использовать ли градиентную потерю
        use_color_consistency (bool): Использовать ли потерю цветовой согласованности
        color_space (str): Цветовое пространство ('lab', 'rgb', 'hsv')
        lambda_nce (float): Вес для PatchNCE потери
        lambda_gradient (float): Вес для градиентной потери
        lambda_consistency (float): Вес для потери цветовой согласованности
    """
    def __init__(
        self,
        feature_dims=[256, 512, 1024],
        patch_sizes=[1, 3, 5],
        patch_counts=[256, 128, 64],
        temperature=0.07,
        use_gradient_loss=True,
        use_color_consistency=True,
        color_space='lab',
        lambda_nce=1.0,
        lambda_gradient=0.5,
        lambda_consistency=0.2
    ):
        super().__init__()
        
        # Основная PatchNCE потеря
        self.patch_nce = MultiScalePatchNCELoss(
            feature_dims=feature_dims,
            patch_sizes=patch_sizes,
            patch_counts=patch_counts,
            temperature=temperature
        )
        
        # Флаги для включения/отключения компонентов
        self.use_gradient_loss = use_gradient_loss
        self.use_color_consistency = use_color_consistency
        
        # Цветовое пространство
        self.color_space = color_space
        
        # Веса для разных компонентов потери
        self.lambda_nce = lambda_nce
        self.lambda_gradient = lambda_gradient
        self.lambda_consistency = lambda_consistency
        
        # Критерии для дополнительных потерь
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
    def color_consistency_loss(self, predicted, target):
        """
        Вычисляет потерю цветовой согласованности между предсказанными и целевыми изображениями.
        
        Args:
            predicted (torch.Tensor): Предсказанное цветное изображение
            target (torch.Tensor): Целевое цветное изображение
            
        Returns:
            torch.Tensor: Значение потери цветовой согласованности
        """
        # Преобразуем в нужное цветовое пространство, если необходимо
        if self.color_space == 'lab':
            # Предполагаем, что входные данные уже в пространстве Lab
            # В пространстве Lab каналы представляют L (яркость), a и b (цветность)
            # Для согласованности цветов нас интересуют только каналы a и b
            predicted_color = predicted[:, 1:, :, :]  # Каналы a и b
            target_color = target[:, 1:, :, :]        # Каналы a и b
        elif self.color_space == 'rgb':
            # Для RGB используем все каналы
            predicted_color = predicted
            target_color = target
        elif self.color_space == 'hsv':
            # Для HSV нас интересуют каналы H (оттенок) и S (насыщенность)
            # Преобразование из RGB в HSV
            # Примечание: это упрощенное преобразование, для точности нужна специальная функция
            predicted_color = predicted[:, :2, :, :]  # Упрощенно: первые два канала
            target_color = target[:, :2, :, :]
        
        # Вычисляем статистики для каждого изображения
        
        # 1. Среднее значение цвета
        pred_mean = torch.mean(predicted_color, dim=[2, 3], keepdim=True)
        target_mean = torch.mean(target_color, dim=[2, 3], keepdim=True)
        mean_loss = self.mse_loss(pred_mean, target_mean)
        
        # 2. Ковариации цветовых каналов
        # Центрируем данные
        pred_centered = predicted_color - pred_mean
        target_centered = target_color - target_mean
        
        # Вычисляем ковариационные матрицы
        batch_size, num_channels = predicted_color.shape[:2]
        pred_cov = torch.zeros(batch_size, num_channels, num_channels, device=predicted_color.device)
        target_cov = torch.zeros(batch_size, num_channels, num_channels, device=predicted_color.device)
        
        for b in range(batch_size):
            pred_flat = pred_centered[b].reshape(num_channels, -1)
            target_flat = target_centered[b].reshape(num_channels, -1)
            
            # Ковариационная матрица
            pred_cov[b] = torch.matmul(pred_flat, pred_flat.transpose(0, 1)) / pred_flat.shape[1]
            target_cov[b] = torch.matmul(target_flat, target_flat.transpose(0, 1)) / target_flat.shape[1]
        
        # Потеря на ковариации
        cov_loss = self.smooth_l1_loss(pred_cov, target_cov)
        
        # 3. Гистограмма цветов
        # Упрощенная версия - мы используем глобальное распределение значений
        # В полной версии здесь был бы расчет гистограмм и их сравнение
        
        # Потеря на распределении значений (используем функцию потери ранговой корреляции)
        pred_sorted, _ = torch.sort(predicted_color.reshape(batch_size, num_channels, -1), dim=2)
        target_sorted, _ = torch.sort(target_color.reshape(batch_size, num_channels, -1), dim=2)
        
        # Используем L1 потерю между отсортированными значениями
        # Это аппроксимирует сравнение распределений
        dist_loss = F.l1_loss(pred_sorted, target_sorted)
        
        # Объединяем все компоненты
        consistency_loss = mean_loss + 0.5 * cov_loss + 0.5 * dist_loss
        
        return consistency_loss
        
    def forward(self, features, predicted, target, attention_maps=None):
        """
        Вычисляет комплексную функцию потерь.
        
        Args:
            features (dict): Словарь с признаками из разных слоев модели
                Ожидается формат: {'encoder': [тензоры-признаков], 'decoder': [тензоры-признаков]}
            predicted (torch.Tensor): Предсказанное цветное изображение [B, C, H, W]
            target (torch.Tensor): Целевое цветное изображение [B, C, H, W]
            attention_maps (list, optional): Список карт внимания для каждого уровня
            
        Returns:
            dict: {
                'total_loss': torch.Tensor,  # Общая потеря
                'nce_loss': torch.Tensor,    # PatchNCE потеря
                'gradient_loss': torch.Tensor,  # Градиентная потеря (если используется)
                'consistency_loss': torch.Tensor  # Потеря цветовой согласованности (если используется)
            }
        """
        # Получаем признаки из энкодера и декодера
        encoder_features = features.get('encoder', [])
        decoder_features = features.get('decoder', [])
        
        # Проверяем, что у нас есть признаки для работы
        assert len(encoder_features) > 0 and len(decoder_features) > 0, "Необходимы признаки из энкодера и декодера"
        
        # Вычисляем PatchNCE потерю между признаками энкодера и декодера
        patch_nce_result = self.patch_nce(encoder_features, decoder_features, attention_maps)
        
        # Получаем потери из результата
        nce_loss = patch_nce_result['nce_loss'] * self.lambda_nce
        gradient_loss = patch_nce_result['gradient_loss'] * self.lambda_gradient if self.use_gradient_loss else 0
        
        # Вычисляем потерю цветовой согласованности, если включена
        consistency_loss = self.color_consistency_loss(predicted, target) * self.lambda_consistency if self.use_color_consistency else 0
        
        # Суммируем все потери
        total_loss = nce_loss
        if self.use_gradient_loss:
            total_loss = total_loss + gradient_loss
        if self.use_color_consistency:
            total_loss = total_loss + consistency_loss
            
        # Формируем словарь с результатами
        result = {
            'total_loss': total_loss,
            'nce_loss': nce_loss,
        }
        
        if self.use_gradient_loss:
            result['gradient_loss'] = gradient_loss
            
        if self.use_color_consistency:
            result['consistency_loss'] = consistency_loss
            
        # Добавляем другие метрики из PatchNCE результата
        result['hard_negative_ratio'] = patch_nce_result['hard_negative_ratio']
        result['layer_losses'] = patch_nce_result['layer_losses']
        
        return result


# Функция для создания MultiScalePatchwiseLoss с параметрами по умолчанию
def create_patch_nce_loss(config=None):
    """
    Создает функцию потерь MultiScalePatchwiseLoss с параметрами по умолчанию или пользовательской конфигурацией.
    
    Args:
        config (dict, optional): Словарь с параметрами конфигурации
        
    Returns:
        MultiScalePatchwiseLoss: Функция потерь
    """
    # Параметры по умолчанию
    default_config = {
        'feature_dims': [256, 512, 1024],
        'patch_sizes': [1, 3, 5],
        'patch_counts': [256, 128, 64],
        'temperature': 0.07,
        'use_gradient_loss': True,
        'use_color_consistency': True,
        'color_space': 'lab',
        'lambda_nce': 1.0,
        'lambda_gradient': 0.5,
        'lambda_consistency': 0.2
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем функцию потерь с указанными параметрами
    loss_fn = MultiScalePatchwiseLoss(**default_config)
    
    return loss_fn


if __name__ == "__main__":
    # Пример использования
    
    # Создаем функцию потерь
    loss_fn = create_patch_nce_loss()
    
    # Создаем тестовые данные
    batch_size = 4
    
    # Признаки из энкодера и декодера
    encoder_features = [
        torch.randn(batch_size, 256, 64, 64),
        torch.randn(batch_size, 512, 32, 32),
        torch.randn(batch_size, 1024, 16, 16)
    ]
    
    decoder_features = [
        torch.randn(batch_size, 256, 64, 64),
        torch.randn(batch_size, 512, 32, 32),
        torch.randn(batch_size, 1024, 16, 16)
    ]
    
    # Предсказанное и целевое изображения (предполагаем пространство Lab)
    predicted = torch.randn(batch_size, 3, 256, 256)
    target = torch.randn(batch_size, 3, 256, 256)
    
    # Вычисляем потери
    features = {
        'encoder': encoder_features,
        'decoder': decoder_features
    }
    
    loss_result = loss_fn(features, predicted, target)
    
    # Выводим результаты
    for key, value in loss_result.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item()}")
        elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
            print(f"{key}: {[v.item() for v in value]}")
        else:
            print(f"{key}: {value}")