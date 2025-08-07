"""
VGG Perceptual Loss: Расширенный перцептивный критерий с механизмом согласованности.

Данный модуль реализует расширенную версию VGG Perceptual Loss для повышения
перцептивного качества колоризации изображений. Он использует предобученную VGG сеть
для извлечения высокоуровневых признаков и сравнения предсказанных цветных 
изображений с эталонными на уровне этих признаков, а не пиксельных значений.

Ключевые особенности:
- Многоуровневые признаки VGG для захвата различных аспектов изображения
- Механизм пространственной согласованности для улучшения цветовой когерентности
- Локальная согласованность стиля для сохранения текстур и узоров
- Адаптивное взвешивание слоев на основе оценки важности каждого уровня
- Расширенные метрики сравнения признаков (LPIPS-подобные)

Преимущества для колоризации:
- Помогает сохранить текстуры и детали оригинального изображения
- Улучшает пространственную когерентность цвета
- Обеспечивает лучшие визуальные результаты, чем простые попиксельные метрики (L1, L2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple
import numpy as np


class VGGEncoder(nn.Module):
    """
    Энкодер на основе VGG для извлечения перцептивных признаков.
    
    Args:
        vgg_type (str): Тип архитектуры VGG ('vgg16', 'vgg19')
        feature_layers (list): Список слоев для извлечения признаков
        use_pretrained (bool): Использовать ли предобученные веса
        requires_grad (bool): Требуется ли обучение весов VGG
        use_input_norm (bool): Применять ли нормализацию входных данных
        mean (list): Средние значения для нормализации
        std (list): Стандартные отклонения для нормализации
    """
    def __init__(
        self,
        vgg_type='vgg19',
        feature_layers=[2, 7, 12, 21, 30],
        use_pretrained=True,
        requires_grad=False,
        use_input_norm=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ):
        super(VGGEncoder, self).__init__()
        
        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm
        
        # Регистрируем среднее и стандартное отклонение как буферы
        if use_input_norm:
            self.register_buffer('mean', torch.Tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor(std).view(1, 3, 1, 1))
        
        # Загружаем модель VGG
        if vgg_type == 'vgg16':
            vgg = models.vgg16(pretrained=use_pretrained).features
        elif vgg_type == 'vgg19':
            vgg = models.vgg19(pretrained=use_pretrained).features
        else:
            raise ValueError(f"Неподдерживаемый тип VGG: {vgg_type}")
        
        # Замораживаем веса, если не требуется обучение
        if not requires_grad:
            for param in vgg.parameters():
                param.requires_grad = False
        
        # Создаем модули для каждого уровня признаков
        vgg_layers = {}
        
        # Индексы конволюционных блоков VGG-19:
        # conv1_1 (0), relu1_1 (1), conv1_2 (2), relu1_2 (3), pool1 (4)
        # conv2_1 (5), relu2_1 (6), conv2_2 (7), relu2_2 (8), pool2 (9)
        # conv3_1 (10), relu3_1 (11), conv3_2 (12), relu3_2 (13), conv3_3 (14), relu3_3 (15), conv3_4 (16), relu3_4 (17), pool3 (18)
        # conv4_1 (19), relu4_1 (20), conv4_2 (21), relu4_2 (22), conv4_3 (23), relu4_3 (24), conv4_4 (25), relu4_4 (26), pool4 (27)
        # conv5_1 (28), relu5_1 (29), conv5_2 (30), relu5_2 (31), conv5_3 (32), relu5_3 (33), conv5_4 (34), relu5_4 (35), pool5 (36)
        
        layer_name_mapping = {
            # VGG-16 and VGG-19 common layers
            0: 'conv1_1', 1: 'relu1_1', 2: 'conv1_2', 3: 'relu1_2', 4: 'pool1',
            5: 'conv2_1', 6: 'relu2_1', 7: 'conv2_2', 8: 'relu2_2', 9: 'pool2',
            10: 'conv3_1', 11: 'relu3_1', 12: 'conv3_2', 13: 'relu3_2',
            # Different for VGG-16 and VGG-19
            14: 'conv3_3' if vgg_type == 'vgg16' else 'conv3_3',
            15: 'relu3_3' if vgg_type == 'vgg16' else 'relu3_3',
            16: 'pool3' if vgg_type == 'vgg16' else 'conv3_4',
            17: 'conv4_1' if vgg_type == 'vgg16' else 'relu3_4',
            18: 'relu4_1' if vgg_type == 'vgg16' else 'pool3',
        }
        
        # Если VGG-19, добавляем оставшиеся слои
        if vgg_type == 'vgg19':
            layer_name_mapping.update({
                19: 'conv4_1', 20: 'relu4_1', 21: 'conv4_2', 22: 'relu4_2',
                23: 'conv4_3', 24: 'relu4_3', 25: 'conv4_4', 26: 'relu4_4', 27: 'pool4',
                28: 'conv5_1', 29: 'relu5_1', 30: 'conv5_2', 31: 'relu5_2',
                32: 'conv5_3', 33: 'relu5_3', 34: 'conv5_4', 35: 'relu5_4', 36: 'pool5'
            })
            
        # Создаем модули для каждого уровня признаков
        for i, layer_idx in enumerate(feature_layers):
            # Создаем последовательность слоев до нужного индекса
            seq = vgg[:layer_idx+1]
            vgg_layers[f'layer{i+1}'] = seq
            
        self.vgg_layers = nn.ModuleDict(vgg_layers)
        
        # Словарь для хранения имен слоев и их индексов
        self.layer_names = {}
        for i, layer_idx in enumerate(feature_layers):
            if layer_idx in layer_name_mapping:
                self.layer_names[f'layer{i+1}'] = layer_name_mapping[layer_idx]
            else:
                self.layer_names[f'layer{i+1}'] = f'layer{layer_idx}'
        
        # Размерности каналов для каждого извлекаемого слоя
        # Для VGG-16 и VGG-19 они соответствуют следующим значениям:
        # conv1_2: 64, conv2_2: 128, conv3_2: 256, conv4_2: 512, conv5_2: 512
        self.channels = []
        for layer_idx in feature_layers:
            if layer_idx <= 4:  # conv1 block
                self.channels.append(64)
            elif layer_idx <= 9:  # conv2 block
                self.channels.append(128)
            elif layer_idx <= 18:  # conv3 block
                self.channels.append(256)
            elif layer_idx <= 27:  # conv4 block
                self.channels.append(512)
            else:  # conv5 block
                self.channels.append(512)
        
    def forward(self, x):
        """
        Прямое распространение для извлечения признаков.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            dict: Словарь с признаками для каждого уровня
                {layer_name: признаки}
        """
        # Нормализуем входные данные, если требуется
        if self.use_input_norm:
            # Если вход в Lab пространстве, преобразуем его в RGB
            if x.size(1) == 3:  # Предполагаем, что уже в RGB или Lab
                if hasattr(self, 'is_lab') and self.is_lab:
                    # Преобразуем из Lab в RGB (упрощенно)
                    # Это очень упрощенное преобразование, для точности нужна специальная функция
                    x = (x + 1) / 2  # Нормализуем в [0, 1]
                
                # Нормализуем по средним и стандартным отклонениям ImageNet
                x = (x - self.mean) / self.std
            else:
                # Для черно-белых изображений повторяем канал трижды
                x = x.repeat(1, 3, 1, 1)
                x = (x - self.mean) / self.std
        
        # Извлекаем признаки для каждого уровня
        features = {}
        for name, layer in self.vgg_layers.items():
            x = layer(x)
            features[name] = x
            
        return features
        
    def get_layer_names(self):
        """
        Возвращает имена слоев VGG, используемые для извлечения признаков.
        
        Returns:
            dict: Словарь {layer_name: vgg_layer_name}
        """
        return self.layer_names
        
    def get_channels(self):
        """
        Возвращает количество каналов для каждого слоя.
        
        Returns:
            list: Список количества каналов для каждого слоя
        """
        return self.channels


class StyleLoss(nn.Module):
    """
    Функция потерь для сохранения стиля (текстуры) изображения.
    
    Args:
        eps (float): Малая константа для численной стабильности
    """
    def __init__(self, eps=1e-8):
        super(StyleLoss, self).__init__()
        self.eps = eps
        
    def gram_matrix(self, x):
        """
        Вычисляет матрицу Грама для тензора признаков.
        
        Args:
            x (torch.Tensor): Тензор признаков [B, C, H, W]
            
        Returns:
            torch.Tensor: Матрица Грама [B, C, C]
        """
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(C * H * W)
    
    def forward(self, predicted, target):
        """
        Вычисляет потерю стиля между предсказанным и целевым изображениями.
        
        Args:
            predicted (torch.Tensor): Признаки предсказанного изображения [B, C, H, W]
            target (torch.Tensor): Признаки целевого изображения [B, C, H, W]
            
        Returns:
            torch.Tensor: Значение функции потерь
        """
        # Вычисляем матрицы Грама
        gram_pred = self.gram_matrix(predicted)
        gram_target = self.gram_matrix(target)
        
        # Вычисляем потерю
        return F.mse_loss(gram_pred, gram_target)


class ContentLoss(nn.Module):
    """
    Функция потерь для сохранения содержимого изображения.
    
    Args:
        loss_type (str): Тип функции потерь ('l1', 'l2', 'smooth_l1', 'lpips_like')
        normalize (bool): Нормализовать ли признаки перед вычислением потерь
    """
    def __init__(self, loss_type='l1', normalize=True):
        super(ContentLoss, self).__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        
        # Инициализируем веса для LPIPS-подобной метрики
        if loss_type == 'lpips_like':
            self.lpips_weights = nn.Parameter(torch.ones(1), requires_grad=True)
            
    def forward(self, predicted, target):
        """
        Вычисляет потерю содержимого между предсказанным и целевым изображениями.
        
        Args:
            predicted (torch.Tensor): Признаки предсказанного изображения [B, C, H, W]
            target (torch.Tensor): Признаки целевого изображения [B, C, H, W]
            
        Returns:
            torch.Tensor: Значение функции потерь
        """
        # Нормализация признаков
        if self.normalize:
            pred_norm = F.normalize(predicted, p=2, dim=1)
            target_norm = F.normalize(target, p=2, dim=1)
        else:
            pred_norm = predicted
            target_norm = target
            
        # Выбор типа функции потерь
        if self.loss_type == 'l1':
            return F.l1_loss(pred_norm, target_norm)
        elif self.loss_type == 'l2':
            return F.mse_loss(pred_norm, target_norm)
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred_norm, target_norm)
        elif self.loss_type == 'lpips_like':
            # LPIPS-подобная метрика: взвешенная сумма расстояний между нормализованными признаками
            diff = (pred_norm - target_norm) ** 2  # Квадратичная разница
            # Пространственное усреднение
            spatial_avg = diff.mean([2, 3])  # [B, C]
            # Взвешенная сумма по каналам
            weighted_avg = (spatial_avg * self.lpips_weights).sum(1).mean()  # Среднее по батчу
            return weighted_avg
        else:
            raise ValueError(f"Неподдерживаемый тип функции потерь: {self.loss_type}")


class SpatialCorrelationLoss(nn.Module):
    """
    Функция потерь для сохранения пространственных корреляций между областями изображения.
    
    Args:
        window_size (int): Размер окна для вычисления локальных корреляций
        stride (int): Шаг скольжения окна
        normalize (bool): Нормализовать ли признаки перед вычислением корреляций
    """
    def __init__(self, window_size=7, stride=3, normalize=True):
        super(SpatialCorrelationLoss, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        
    def compute_spatial_correlation(self, x, window_size, stride):
        """
        Вычисляет матрицу пространственных корреляций между патчами.
        
        Args:
            x (torch.Tensor): Тензор признаков [B, C, H, W]
            window_size (int): Размер окна
            stride (int): Шаг скольжения окна
            
        Returns:
            torch.Tensor: Матрица пространственных корреляций [B, N, N]
                где N - количество патчей
        """
        B, C, H, W = x.shape
        
        # Извлекаем патчи с помощью unfold
        patches = F.unfold(x, kernel_size=window_size, stride=stride)
        # Размер: [B, C*window_size*window_size, L]
        # где L - количество патчей
        
        # Изменяем форму для вычисления корреляций
        patches = patches.transpose(1, 2).reshape(B, -1, C, window_size*window_size)
        # Размер: [B, L, C, window_size*window_size]
        
        # Сглаживаем размерности патча
        patches = patches.reshape(B, patches.size(1), -1)
        # Размер: [B, L, C*window_size*window_size]
        
        # Нормализуем патчи, если требуется
        if self.normalize:
            patches = F.normalize(patches, p=2, dim=2)
            
        # Вычисляем матрицу корреляций между всеми парами патчей
        correlations = torch.bmm(patches, patches.transpose(1, 2))
        # Размер: [B, L, L]
        
        return correlations
    
    def forward(self, predicted, target):
        """
        Вычисляет потерю пространственных корреляций.
        
        Args:
            predicted (torch.Tensor): Признаки предсказанного изображения [B, C, H, W]
            target (torch.Tensor): Признаки целевого изображения [B, C, H, W]
            
        Returns:
            torch.Tensor: Значение функции потерь
        """
        # Вычисляем матрицы пространственных корреляций
        pred_corr = self.compute_spatial_correlation(predicted, self.window_size, self.stride)
        target_corr = self.compute_spatial_correlation(target, self.window_size, self.stride)
        
        # Вычисляем потерю как среднеквадратическую разницу между матрицами корреляций
        return F.mse_loss(pred_corr, target_corr)


class ColorConsistencyLoss(nn.Module):
    """
    Функция потерь для сохранения цветовой согласованности между семантически похожими областями.
    
    Args:
        similarity_threshold (float): Порог сходства для определения семантически похожих областей
        color_space (str): Цветовое пространство для сравнения ('rgb', 'lab')
    """
    def __init__(self, similarity_threshold=0.7, color_space='lab'):
        super(ColorConsistencyLoss, self).__init__()
        self.similarity_threshold = similarity_threshold
        self.color_space = color_space
        
    def forward(self, predicted, target, semantic_features):
        """
        Вычисляет потерю цветовой согласованности.
        
        Args:
            predicted (torch.Tensor): Предсказанное цветное изображение [B, C, H, W]
            target (torch.Tensor): Целевое цветное изображение [B, C, H, W]
            semantic_features (torch.Tensor): Семантические признаки [B, D, H, W]
            
        Returns:
            torch.Tensor: Значение функции потерь
        """
        B, C, H, W = predicted.shape
        
        # Извлекаем цветовую информацию
        if self.color_space == 'lab':
            # В пространстве Lab: каналы a и b содержат цветовую информацию
            pred_color = predicted[:, 1:, :, :]  # [B, 2, H, W]
            target_color = target[:, 1:, :, :]   # [B, 2, H, W]
        else:  # 'rgb'
            # В пространстве RGB используем все каналы
            pred_color = predicted
            target_color = target
        
        # Изменяем форму для пиксельных операций
        pred_color = pred_color.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        target_color = target_color.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # Изменяем форму семантических признаков
        D = semantic_features.size(1)
        semantic_flat = semantic_features.view(B, D, -1).transpose(1, 2)  # [B, H*W, D]
        
        # Нормализуем семантические признаки для вычисления сходства
        semantic_norm = F.normalize(semantic_flat, p=2, dim=2)
        
        # Инициализируем потерю
        loss = 0.0
        
        # Для каждого изображения в батче
        for b in range(B):
            # Вычисляем матрицу семантического сходства между всеми пикселями
            similarity = torch.mm(semantic_norm[b], semantic_norm[b].transpose(0, 1))  # [H*W, H*W]
            
            # Находим семантически похожие пары пикселей
            similar_pairs = torch.nonzero(similarity > self.similarity_threshold, as_tuple=False)
            
            # Если есть похожие пары
            if similar_pairs.size(0) > 0:
                # Для каждой пары похожих пикселей
                idx1, idx2 = similar_pairs[:, 0], similar_pairs[:, 1]
                
                # Получаем цвета для этих пикселей
                pred_color1, pred_color2 = pred_color[b, idx1], pred_color[b, idx2]
                target_color1, target_color2 = target_color[b, idx1], target_color[b, idx2]
                
                # Вычисляем разницу между предсказанными цветами
                pred_diff = pred_color1 - pred_color2
                # Вычисляем разницу между целевыми цветами
                target_diff = target_color1 - target_color2
                
                # Потеря: предсказанная разница должна быть близка к целевой разнице
                pair_loss = F.mse_loss(pred_diff, target_diff)
                
                # Добавляем к общей потере
                loss += pair_loss
                
        # Нормализуем потерю по количеству изображений в батче
        loss = loss / max(B, 1)
        
        return loss


class VGGPerceptualLoss(nn.Module):
    """
    Расширенная VGG Perceptual Loss для колоризации изображений.
    
    Args:
        vgg_type (str): Тип архитектуры VGG ('vgg16', 'vgg19')
        feature_layers (list): Список слоев для извлечения признаков
        layer_weights (list): Веса для каждого слоя
        content_weight (float): Вес для потери содержимого
        style_weight (float): Вес для потери стиля
        correlation_weight (float): Вес для потери пространственной корреляции
        consistency_weight (float): Вес для потери цветовой согласованности
        content_loss_type (str): Тип потери содержимого
        use_adaptive_weights (bool): Использовать ли адаптивные веса для слоев
    """
    def __init__(
        self,
        vgg_type='vgg19',
        feature_layers=[2, 7, 12, 21, 30],
        layer_weights=None,
        content_weight=1.0,
        style_weight=0.5,
        correlation_weight=0.3,
        consistency_weight=0.2,
        content_loss_type='l1',
        use_adaptive_weights=True
    ):
        super(VGGPerceptualLoss, self).__init__()
        
        # Инициализируем VGG энкодер
        self.encoder = VGGEncoder(
            vgg_type=vgg_type,
            feature_layers=feature_layers,
            use_pretrained=True,
            requires_grad=False,
            use_input_norm=True
        )
        
        # Получаем количество слоев
        self.num_layers = len(feature_layers)
        
        # Устанавливаем веса для каждого слоя
        if layer_weights is None:
            # По умолчанию: больший вес для средних слоев, меньший для низкоуровневых и высокоуровневых
            weights = np.ones(self.num_layers) / self.num_layers
            if self.num_layers >= 3:
                weights = np.array([0.5, 1.0, 1.5, 1.0, 0.5][:self.num_layers])
                weights = weights / weights.sum()
            self.layer_weights = nn.Parameter(torch.Tensor(weights), requires_grad=use_adaptive_weights)
        else:
            assert len(layer_weights) == self.num_layers, "Количество весов должно совпадать с количеством слоев"
            weights = np.array(layer_weights) / np.sum(layer_weights)
            self.layer_weights = nn.Parameter(torch.Tensor(weights), requires_grad=use_adaptive_weights)
        
        # Веса для разных типов потерь
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.correlation_weight = correlation_weight
        self.consistency_weight = consistency_weight
        
        # Инициализируем функции потерь
        self.content_loss = ContentLoss(loss_type=content_loss_type)
        self.style_loss = StyleLoss()
        self.correlation_loss = SpatialCorrelationLoss()
        self.consistency_loss = ColorConsistencyLoss()
        
        # Флаг для адаптивных весов
        self.use_adaptive_weights = use_adaptive_weights
        
    def get_normalized_weights(self):
        """
        Получает нормализованные веса для каждого слоя.
        
        Returns:
            torch.Tensor: Нормализованные веса
        """
        return F.softmax(self.layer_weights, dim=0)
    
    def forward(self, predicted, target, grayscale=None):
        """
        Вычисляет перцептивную функцию потерь между предсказанным и целевым изображениями.
        
        Args:
            predicted (torch.Tensor): Предсказанное цветное изображение [B, C, H, W]
            target (torch.Tensor): Целевое цветное изображение [B, C, H, W]
            grayscale (torch.Tensor, optional): Исходное черно-белое изображение [B, 1, H, W]
            
        Returns:
            dict: Словарь с результатами:
                - total_loss: Общая потеря
                - content_loss: Потеря содержимого
                - style_loss: Потеря стиля
                - correlation_loss: Потеря пространственной корреляции
                - consistency_loss: Потеря цветовой согласованности
                - layer_losses: Потери для каждого слоя
        """
        # Получаем нормализованные веса
        if self.use_adaptive_weights:
            norm_weights = self.get_normalized_weights()
        else:
            norm_weights = self.layer_weights
        
        # Извлекаем признаки VGG для предсказанного и целевого изображений
        pred_features = self.encoder(predicted)
        target_features = self.encoder(target)
        
        # Если предоставлено исходное черно-белое изображение, извлекаем его признаки
        if grayscale is not None:
            gray_features = self.encoder(grayscale)
        else:
            gray_features = None
        
        # Инициализируем потери
        total_content_loss = 0.0
        total_style_loss = 0.0
        total_correlation_loss = 0.0
        layer_losses = []
        
        # Вычисляем потери для каждого слоя
        for i in range(self.num_layers):
            layer_name = f'layer{i+1}'
            
            # Получаем признаки текущего слоя
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            
            # Вычисляем потерю содержимого
            content_loss = self.content_loss(pred_feat, target_feat)
            
            # Вычисляем потерю стиля
            style_loss = self.style_loss(pred_feat, target_feat)
            
            # Вычисляем потерю пространственной корреляции
            correlation_loss = self.correlation_loss(pred_feat, target_feat)
            
            # Взвешенная сумма потерь для текущего слоя
            layer_loss = (
                self.content_weight * content_loss +
                self.style_weight * style_loss +
                self.correlation_weight * correlation_loss
            )
            
            # Применяем вес слоя
            weighted_loss = norm_weights[i] * layer_loss
            layer_losses.append(weighted_loss)
            
            # Обновляем общие потери
            total_content_loss += norm_weights[i] * content_loss
            total_style_loss += norm_weights[i] * style_loss
            total_correlation_loss += norm_weights[i] * correlation_loss
        
        # Вычисляем потерю цветовой согласованности, если есть промежуточные признаки
        consistency_loss = 0.0
        if gray_features is not None and self.consistency_weight > 0:
            # Используем признаки среднего слоя для семантического сходства
            mid_layer = self.num_layers // 2
            mid_layer_name = f'layer{mid_layer+1}'
            semantic_features = gray_features[mid_layer_name]
            
            consistency_loss = self.consistency_loss(predicted, target, semantic_features)
            
        # Суммируем все типы потерь
        total_loss = (
            self.content_weight * total_content_loss +
            self.style_weight * total_style_loss +
            self.correlation_weight * total_correlation_loss +
            self.consistency_weight * consistency_loss
        )
        
        # Создаем словарь с результатами
        results = {
            'total_loss': total_loss,
            'content_loss': total_content_loss,
            'style_loss': total_style_loss,
            'correlation_loss': total_correlation_loss,
            'consistency_loss': consistency_loss,
            'layer_losses': layer_losses,
            'layer_weights': norm_weights.detach()
        }
        
        return results


class EnhancedVGGPerceptual(nn.Module):
    """
    Расширенная перцептивная функция потерь с дополнительными компонентами
    для улучшения качества колоризации.
    
    Args:
        vgg_type (str): Тип архитектуры VGG ('vgg16', 'vgg19')
        feature_layers (list): Список слоев для извлечения признаков
        include_pixel_loss (bool): Включать ли попиксельную потерю
        include_gradient_loss (bool): Включать ли градиентную потерю
        include_frequency_loss (bool): Включать ли частотную потерю
        pixel_weight (float): Вес для попиксельной потери
        gradient_weight (float): Вес для градиентной потери
        frequency_weight (float): Вес для частотной потери
        perceptual_weight (float): Вес для перцептивной потери
        pixel_loss_type (str): Тип попиксельной потери ('l1', 'l2', 'smooth_l1')
        use_adaptive_weights (bool): Использовать ли адаптивные веса
    """
    def __init__(
        self,
        vgg_type='vgg19',
        feature_layers=[2, 7, 12, 21, 30],
        include_pixel_loss=True,
        include_gradient_loss=True,
        include_frequency_loss=True,
        pixel_weight=1.0,
        gradient_weight=0.5,
        frequency_weight=0.1,
        perceptual_weight=1.0,
        pixel_loss_type='l1',
        use_adaptive_weights=True
    ):
        super(EnhancedVGGPerceptual, self).__init__()
        
        # Инициализируем базовую перцептивную потерю
        self.perceptual_loss = VGGPerceptualLoss(
            vgg_type=vgg_type,
            feature_layers=feature_layers,
            use_adaptive_weights=use_adaptive_weights
        )
        
        # Флаги для включения/отключения компонентов
        self.include_pixel_loss = include_pixel_loss
        self.include_gradient_loss = include_gradient_loss
        self.include_frequency_loss = include_frequency_loss
        
        # Веса для разных компонентов
        self.pixel_weight = pixel_weight
        self.gradient_weight = gradient_weight
        self.frequency_weight = frequency_weight
        self.perceptual_weight = perceptual_weight
        
        # Тип попиксельной потери
        self.pixel_loss_type = pixel_loss_type
        
        # Инициализируем дополнительные функции потерь
        if include_pixel_loss:
            if pixel_loss_type == 'l1':
                self.pixel_loss_fn = nn.L1Loss()
            elif pixel_loss_type == 'l2':
                self.pixel_loss_fn = nn.MSELoss()
            elif pixel_loss_type == 'smooth_l1':
                self.pixel_loss_fn = nn.SmoothL1Loss()
            else:
                raise ValueError(f"Неподдерживаемый тип попиксельной потери: {pixel_loss_type}")
                
    def gradient_loss(self, predicted, target):
        """
        Вычисляет градиентную потерю между предсказанным и целевым изображениями.
        
        Args:
            predicted (torch.Tensor): Предсказанное изображение [B, C, H, W]
            target (torch.Tensor): Целевое изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Значение градиентной потери
        """
        # Вычисляем градиенты по X и Y для предсказанного изображения
        pred_grad_x = predicted[:, :, :, 1:] - predicted[:, :, :, :-1]
        pred_grad_y = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
        
        # Вычисляем градиенты по X и Y для целевого изображения
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Вычисляем потери для градиентов
        loss_grad_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_grad_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        # Суммируем потери
        return loss_grad_x + loss_grad_y
    
    def frequency_loss(self, predicted, target):
        """
        Вычисляет частотную потерю между предсказанным и целевым изображениями.
        
        Args:
            predicted (torch.Tensor): Предсказанное изображение [B, C, H, W]
            target (torch.Tensor): Целевое изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Значение частотной потери
        """
        # Вычисляем FFT для каждого канала
        batch_size, channels = predicted.shape[:2]
        loss = 0.0
        
        for c in range(channels):
            # Извлекаем канал
            pred_channel = predicted[:, c]
            target_channel = target[:, c]
            
            # Вычисляем FFT
            pred_fft = torch.fft.fft2(pred_channel)
            target_fft = torch.fft.fft2(target_channel)
            
            # Вычисляем амплитудные спектры
            pred_magnitude = torch.abs(pred_fft)
            target_magnitude = torch.abs(target_fft)
            
            # Вычисляем потерю в частотной области (MSE)
            channel_loss = F.mse_loss(pred_magnitude, target_magnitude)
            loss += channel_loss
            
        # Усредняем по каналам
        return loss / channels
    
    def forward(self, predicted, target, grayscale=None):
        """
        Вычисляет расширенную перцептивную функцию потерь.
        
        Args:
            predicted (torch.Tensor): Предсказанное изображение [B, C, H, W]
            target (torch.Tensor): Целевое изображение [B, C, H, W]
            grayscale (torch.Tensor, optional): Исходное черно-белое изображение [B, 1, H, W]
            
        Returns:
            dict: Словарь с результатами:
                - total_loss: Общая потеря
                - perceptual_loss: Перцептивная потеря
                - pixel_loss: Попиксельная потеря (если включена)
                - gradient_loss: Градиентная потеря (если включена)
                - frequency_loss: Частотная потеря (если включена)
                - detailed_losses: Подробные результаты перцептивной потери
        """
        # Вычисляем базовую перцептивную потерю
        perceptual_results = self.perceptual_loss(predicted, target, grayscale)
        perceptual_loss = perceptual_results['total_loss']
        
        # Инициализируем дополнительные потери
        pixel_loss = 0.0
        gradient_loss = 0.0
        frequency_loss = 0.0
        
        # Вычисляем попиксельную потерю, если включена
        if self.include_pixel_loss:
            pixel_loss = self.pixel_loss_fn(predicted, target)
            
        # Вычисляем градиентную потерю, если включена
        if self.include_gradient_loss:
            gradient_loss = self.gradient_loss(predicted, target)
            
        # Вычисляем частотную потерю, если включена
        if self.include_frequency_loss:
            frequency_loss = self.frequency_loss(predicted, target)
            
        # Суммируем все потери с учетом весов
        total_loss = (
            self.perceptual_weight * perceptual_loss +
            self.pixel_weight * pixel_loss +
            self.gradient_weight * gradient_loss +
            self.frequency_weight * frequency_loss
        )
        
        # Создаем словарь с результатами
        results = {
            'total_loss': total_loss,
            'perceptual_loss': perceptual_loss,
            'pixel_loss': pixel_loss,
            'gradient_loss': gradient_loss,
            'frequency_loss': frequency_loss,
            'detailed_losses': perceptual_results
        }
        
        return results


# Функция для создания EnhancedVGGPerceptual с параметрами по умолчанию
def create_vgg_perceptual_loss(config=None):
    """
    Создает функцию потерь EnhancedVGGPerceptual с параметрами по умолчанию 
    или пользовательской конфигурацией.
    
    Args:
        config (dict, optional): Словарь с параметрами конфигурации
        
    Returns:
        EnhancedVGGPerceptual: Функция потерь
    """
    # Параметры по умолчанию
    default_config = {
        'vgg_type': 'vgg19',
        'feature_layers': [2, 7, 12, 21, 30],
        'include_pixel_loss': True,
        'include_gradient_loss': True,
        'include_frequency_loss': True,
        'pixel_weight': 1.0,
        'gradient_weight': 0.5,
        'frequency_weight': 0.1,
        'perceptual_weight': 1.0,
        'pixel_loss_type': 'l1',
        'use_adaptive_weights': True
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем функцию потерь с указанными параметрами
    loss_fn = EnhancedVGGPerceptual(**default_config)
    
    return loss_fn


if __name__ == "__main__":
    # Пример использования
    
    # Создаем функцию потерь
    loss_fn = create_vgg_perceptual_loss()
    
    # Создаем тестовые данные
    batch_size = 2
    
    # Предсказанное и целевое изображения (предполагаем пространство RGB)
    predicted = torch.rand(batch_size, 3, 256, 256)
    target = torch.rand(batch_size, 3, 256, 256)
    
    # Исходное черно-белое изображение
    grayscale = torch.rand(batch_size, 1, 256, 256)
    
    # Вычисляем потери
    loss_result = loss_fn(predicted, target, grayscale)
    
    # Выводим результаты
    for key, value in loss_result.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item()}")
        elif isinstance(value, dict):
            print(f"{key}: {', '.join([f'{k}: {v.item() if isinstance(v, torch.Tensor) else v}' for k, v in value.items() if not isinstance(v, list)])}")
        else:
            print(f"{key}: {value}")