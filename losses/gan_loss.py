"""
GAN Loss: Функции потерь для генеративно-состязательного обучения с системой наград и наказаний.

Данный модуль реализует специализированные функции потерь для GAN-подхода к колоризации
изображений, включая систему "наград и наказаний" для улучшения качества и реализма
генерируемых цветов. Модуль обеспечивает "мотивацию" для генератора создавать
реалистичные цветные изображения, а для дискриминатора - точно определять качество колоризации.

Ключевые особенности:
- Классические и современные GAN-функции потерь (vanilla, LSGAN, WGAN, hinge)
- Реализация системы "наград" для поощрения реалистичной колоризации
- Механизм "наказаний" для исправления типичных ошибок (неестественные цвета, размытие)
- Взвешенный компонент для управления балансом потерь генератора и дискриминатора
- Динамическая настройка интенсивности обратной связи в зависимости от производительности

Преимущества для колоризации:
- Улучшает общий реализм и естественность цветов
- Помогает избегать "безопасных" но скучных бледных решений
- Создает эффект обучения с "мотивацией", как если бы модель стремилась к одобрению
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GANLoss(nn.Module):
    """
    Общая функция потерь GAN с различными режимами работы.
    
    Args:
        gan_mode (str): Режим работы GAN ('vanilla', 'lsgan', 'wgan', 'hinge')
        target_real_label (float): Целевое значение для реальных изображений
        target_fake_label (float): Целевое значение для сгенерированных изображений
        label_smoothing (float): Коэффициент сглаживания меток (для vanilla и lsgan)
        use_reward_penalty (bool): Использовать ли систему наград и наказаний
    """
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0, 
                 label_smoothing=0.1, use_reward_penalty=True):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.label_smoothing = label_smoothing
        self.use_reward_penalty = use_reward_penalty
        
        # История потерь для отслеживания прогресса
        self.history_G_losses = []
        self.history_D_losses = []
        
        # Множитель для наград/наказаний, адаптируется в процессе обучения
        self.reward_multiplier = 1.0
        self.max_reward = 2.0  # Максимальный множитель награды
        self.min_reward = 0.5  # Минимальный множитель награды
        
        # Инициализация потерь в зависимости от режима GAN
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode in ['wgan', 'hinge']:
            self.loss = None
        else:
            raise ValueError(f'Неподдерживаемый режим GAN: {gan_mode}')
            
    def get_target_tensor(self, prediction, target_is_real):
        """
        Создает целевой тензор для функции потерь.
        
        Args:
            prediction (torch.Tensor): Выход дискриминатора
            target_is_real (bool): Цель - реальное изображение или сгенерированное
            
        Returns:
            torch.Tensor: Целевой тензор для обучения
        """
        if target_is_real:
            if self.real_label_tensor is None or self.real_label_tensor.numel() != prediction.numel():
                # Применяем сглаживание меток для реальных изображений для улучшения стабильности
                if self.label_smoothing > 0:
                    # Сглаживаем метки в сторону 0.5 для реальных изображений
                    smooth_real_value = self.real_label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                    self.real_label_tensor = torch.full_like(prediction, smooth_real_value)
                else:
                    self.real_label_tensor = torch.full_like(prediction, self.real_label)
                    
            target_tensor = self.real_label_tensor
        else:
            if self.fake_label_tensor is None or self.fake_label_tensor.numel() != prediction.numel():
                self.fake_label_tensor = torch.full_like(prediction, self.fake_label)
            target_tensor = self.fake_label_tensor
            
        return target_tensor
    
    def apply_reward_penalty(self, loss, quality_score=None, target_is_real=True, is_generator=True):
        """
        Применяет систему наград и наказаний к функции потерь на основе качества изображения.
        
        Args:
            loss (torch.Tensor): Исходная функция потерь
            quality_score (torch.Tensor, optional): Оценка качества изображения [0, 1]
            target_is_real (bool): Является ли цель реальным изображением
            is_generator (bool): Является ли это потерей для генератора
            
        Returns:
            torch.Tensor: Модифицированная функция потерь
        """
        if not self.use_reward_penalty or quality_score is None:
            return loss
        
        # Преобразуем скаляр в тензор, если необходимо
        if isinstance(quality_score, (float, int)):
            quality_score = torch.tensor(quality_score).to(loss.device)
            
        # Базовый множитель для наград/наказаний
        multiplier = self.reward_multiplier
        
        # Для генератора
        if is_generator:
            if target_is_real:  # Генератор хочет, чтобы его выход выглядел реальным
                # Награда: увеличиваем потерю для низкокачественных результатов,
                # чтобы генератор сильнее стремился к улучшению
                reward = multiplier * (1.0 - quality_score) + 1.0
                
                # Ограничиваем максимальную награду
                reward = torch.clamp(reward, 1.0, self.max_reward)
                
                return loss * reward
            else:
                # Этот случай обычно не используется для генератора
                return loss
        # Для дискриминатора
        else:
            if target_is_real:  # Дискриминатор оценивает реальные изображения
                # Стандартная потеря для реальных изображений
                return loss
            else:  # Дискриминатор оценивает поддельные изображения
                # Наказание: уменьшаем потерю для высококачественных подделок,
                # чтобы дискриминатор был менее уверен в их фальшивости
                penalty = multiplier * quality_score + self.min_reward
                
                # Ограничиваем минимальное наказание
                penalty = torch.clamp(penalty, self.min_reward, 1.0)
                
                return loss * penalty
    
    def update_reward_multiplier(self, G_loss_mean, D_loss_mean):
        """
        Обновляет множитель награды на основе истории потерь.
        
        Args:
            G_loss_mean (float): Среднее значение потери генератора
            D_loss_mean (float): Среднее значение потери дискриминатора
        """
        # Добавляем потери в историю
        self.history_G_losses.append(G_loss_mean)
        self.history_D_losses.append(D_loss_mean)
        
        # Ограничиваем размер истории
        max_history = 100
        if len(self.history_G_losses) > max_history:
            self.history_G_losses = self.history_G_losses[-max_history:]
            self.history_D_losses = self.history_D_losses[-max_history:]
            
        # Если история слишком короткая, не обновляем множитель
        if len(self.history_G_losses) < 10:
            return
        
        # Вычисляем средние значения потерь за последние итерации
        recent_G_mean = np.mean(self.history_G_losses[-10:])
        recent_D_mean = np.mean(self.history_D_losses[-10:])
        
        # Вычисляем отношение потерь G к D
        ratio = recent_G_mean / max(recent_D_mean, 1e-8)
        
        # Если генератор сильно отстает от дискриминатора, увеличиваем награды
        if ratio > 5.0:
            self.reward_multiplier = min(self.reward_multiplier * 1.05, 2.0)
        # Если дискриминатор отстает, уменьшаем награды
        elif ratio < 0.2:
            self.reward_multiplier = max(self.reward_multiplier * 0.95, 0.5)
    
    def get_reward_stats(self):
        """
        Возвращает текущие статистики системы наград.
        
        Returns:
            dict: Статистики системы наград
        """
        return {
            'reward_multiplier': self.reward_multiplier,
            'G_loss_history': self.history_G_losses[-10:] if len(self.history_G_losses) >= 10 else self.history_G_losses,
            'D_loss_history': self.history_D_losses[-10:] if len(self.history_D_losses) >= 10 else self.history_D_losses
        }
            
    def __call__(self, prediction, target_is_real, quality_score=None, is_generator=True):
        """
        Вычисляет GAN-функцию потерь.
        
        Args:
            prediction (torch.Tensor): Выход дискриминатора
            target_is_real (bool): Цель - реальное изображение или сгенерированное
            quality_score (torch.Tensor, optional): Оценка качества изображения [0, 1]
            is_generator (bool): Является ли это потерей для генератора
            
        Returns:
            torch.Tensor: Значение функции потерь
        """
        # Определяем режим работы GAN
        if self.gan_mode == 'vanilla':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'lsgan':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if is_generator:
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            else:  # Дискриминатор
                if target_is_real:
                    loss = F.relu(1.0 - prediction).mean()
                else:
                    loss = F.relu(1.0 + prediction).mean()
        
        # Применяем систему наград и наказаний
        if self.use_reward_penalty and quality_score is not None:
            loss = self.apply_reward_penalty(loss, quality_score, target_is_real, is_generator)
            
        return loss


class AdversarialColorLoss(nn.Module):
    """
    Специализированный GAN-loss для задачи колоризации с акцентом на цветовые характеристики.
    
    Args:
        mode (str): Режим работы ('global', 'patchgan', 'multiscale')
        color_space (str): Цветовое пространство для анализа ('lab', 'rgb', 'hsv')
        color_weight (float): Вес для потери цветовых характеристик
        texture_weight (float): Вес для потери текстурных характеристик
        semantic_weight (float): Вес для потери семантических характеристик
        gan_mode (str): Режим работы GAN ('vanilla', 'lsgan', 'wgan', 'hinge')
    """
    def __init__(self, mode='patchgan', color_space='lab', 
                 color_weight=1.0, texture_weight=0.7, semantic_weight=0.3,
                 gan_mode='lsgan'):
        super(AdversarialColorLoss, self).__init__()
        
        self.mode = mode
        self.color_space = color_space
        
        # Веса для разных компонентов
        self.color_weight = color_weight
        self.texture_weight = texture_weight
        self.semantic_weight = semantic_weight
        
        # Базовая GAN-функция потерь
        self.gan_loss = GANLoss(gan_mode=gan_mode, use_reward_penalty=True)
        
        # Функция потери цвета (фокусируется на цветовых каналах)
        self.color_loss = nn.L1Loss()
        
        # Регистрируем счетчики для мониторинга
        self.register_buffer('realistic_count', torch.tensor(0))
        self.register_buffer('total_count', torch.tensor(0))
        
        # Метрики оценки производительности
        self.register_buffer('color_errors', torch.tensor(0.0))
        self.register_buffer('texture_errors', torch.tensor(0.0))
        self.register_buffer('semantic_errors', torch.tensor(0.0))
        
    def extract_color_features(self, img):
        """
        Извлекает цветовые характеристики из изображения.
        
        Args:
            img (torch.Tensor): Изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Цветовые характеристики
        """
        if self.color_space == 'lab':
            # Предполагаем, что входное изображение уже в пространстве Lab
            # Извлекаем только цветовые каналы a и b
            if img.size(1) >= 3:
                return img[:, 1:3, :, :]
            else:
                raise ValueError("Для извлечения цветовых характеристик в пространстве Lab необходимо 3 канала")
        elif self.color_space == 'rgb':
            # В RGB пространстве используем все каналы
            return img
        elif self.color_space == 'hsv':
            # Для HSV пространства нам нужно преобразовать изображение
            # Это упрощенная версия, для точного преобразования требуется специальная функция
            # Предполагаем, что входное изображение в RGB
            if img.size(1) == 3:
                # Извлекаем только каналы H и S (приближение)
                r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
                max_rgb, _ = torch.max(img, dim=1, keepdim=True)
                min_rgb, _ = torch.min(img, dim=1, keepdim=True)
                delta = max_rgb - min_rgb + 1e-8
                
                # Приближение для H
                h = torch.zeros_like(r)
                mask_r = (max_rgb == r)
                mask_g = (max_rgb == g)
                mask_b = (max_rgb == b)
                h[mask_r] = (g - b)[mask_r] / delta[mask_r]
                h[mask_g] = 2.0 + (b - r)[mask_g] / delta[mask_g]
                h[mask_b] = 4.0 + (r - g)[mask_b] / delta[mask_b]
                h = h / 6.0
                
                # Приближение для S
                s = delta / (max_rgb + 1e-8)
                
                return torch.cat([h, s], dim=1)
            else:
                raise ValueError("Для извлечения цветовых характеристик в пространстве HSV необходимо 3 канала")
        else:
            raise ValueError(f"Неподдерживаемое цветовое пространство: {self.color_space}")
            
    def compute_color_quality_score(self, fake_color, real_color):
        """
        Вычисляет оценку качества цвета на основе сравнения сгенерированного
        и реального цвета.
        
        Args:
            fake_color (torch.Tensor): Сгенерированные цветовые характеристики
            real_color (torch.Tensor): Реальные цветовые характеристики
            
        Returns:
            torch.Tensor: Оценка качества цвета [0, 1]
        """
        # Вычисляем среднюю ошибку цвета
        color_error = F.l1_loss(fake_color, real_color, reduction='none').mean(dim=[1, 2, 3])
        
        # Преобразуем ошибку в оценку качества (меньше ошибка - выше качество)
        # Используем экспоненциальное масштабирование для нелинейного отображения
        quality_score = torch.exp(-5 * color_error)  # Масштабирующий фактор 5 подобран эмпирически
        
        # Обновляем счетчик ошибок цвета
        self.color_errors += color_error.mean().item()
        
        return quality_score
    
    def compute_texture_consistency(self, fake_img, real_img):
        """
        Вычисляет согласованность текстуры между сгенерированным и реальным изображениями.
        
        Args:
            fake_img (torch.Tensor): Сгенерированное изображение
            real_img (torch.Tensor): Реальное изображение
            
        Returns:
            torch.Tensor: Оценка согласованности текстуры [0, 1]
        """
        # Вычисляем градиенты по X и Y для обоих изображений
        def gradient(img):
            # Градиент по X
            grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
            # Градиент по Y
            grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
            return grad_x, grad_y
        
        fake_grad_x, fake_grad_y = gradient(fake_img)
        real_grad_x, real_grad_y = gradient(real_img)
        
        # Вычисляем ошибки градиентов
        grad_x_error = F.l1_loss(fake_grad_x, real_grad_x, reduction='none').mean(dim=[1, 2, 3])
        grad_y_error = F.l1_loss(fake_grad_y, real_grad_y, reduction='none').mean(dim=[1, 2, 3])
        
        # Вычисляем среднюю ошибку градиентов
        grad_error = (grad_x_error + grad_y_error) / 2.0
        
        # Преобразуем ошибку в оценку согласованности
        texture_score = torch.exp(-5 * grad_error)
        
        # Обновляем счетчик ошибок текстуры
        self.texture_errors += grad_error.mean().item()
        
        return texture_score
    
    def compute_semantic_consistency(self, fake_features, real_features):
        """
        Вычисляет семантическую согласованность между сгенерированными и реальными признаками.
        
        Args:
            fake_features (list): Список сгенерированных признаков
            real_features (list): Список реальных признаков
            
        Returns:
            torch.Tensor: Оценка семантической согласованности [0, 1]
        """
        if fake_features is None or real_features is None or len(fake_features) == 0:
            # Если признаки не предоставлены, возвращаем нейтральную оценку
            return torch.ones(fake_img.size(0), device=fake_img.device) * 0.5
        
        # Вычисляем косинусное сходство между признаками
        similarities = []
        
        for fake_feat, real_feat in zip(fake_features, real_features):
            # Нормализуем признаки
            fake_norm = F.normalize(fake_feat, p=2, dim=1)
            real_norm = F.normalize(real_feat, p=2, dim=1)
            
            # Вычисляем косинусное сходство
            similarity = torch.sum(fake_norm * real_norm, dim=1)
            
            # Масштабируем в диапазон [0, 1]
            similarity = (similarity + 1) / 2
            
            similarities.append(similarity)
        
        # Усредняем сходство по всем признакам
        semantic_score = torch.stack(similarities).mean(dim=0)
        
        # Обновляем счетчик ошибок семантики
        semantic_error = 1.0 - semantic_score.mean().item()
        self.semantic_errors += semantic_error
        
        return semantic_score
            
    def compute_quality_score(self, fake_img, real_img, fake_features=None, real_features=None):
        """
        Вычисляет общую оценку качества изображения, учитывая цвет, текстуру и семантику.
        
        Args:
            fake_img (torch.Tensor): Сгенерированное изображение
            real_img (torch.Tensor): Реальное изображение
            fake_features (list, optional): Список сгенерированных признаков
            real_features (list, optional): Список реальных признаков
            
        Returns:
            torch.Tensor: Общая оценка качества [0, 1]
        """
        # Извлекаем цветовые характеристики
        fake_color = self.extract_color_features(fake_img)
        real_color = self.extract_color_features(real_img)
        
        # Вычисляем оценку качества цвета
        color_score = self.compute_color_quality_score(fake_color, real_color)
        
        # Вычисляем согласованность текстуры
        texture_score = self.compute_texture_consistency(fake_img, real_img)
        
        # Вычисляем семантическую согласованность
        semantic_score = self.compute_semantic_consistency(fake_features, real_features)
        
        # Объединяем оценки с весами
        quality_score = (
            self.color_weight * color_score +
            self.texture_weight * texture_score +
            self.semantic_weight * semantic_score
        ) / (self.color_weight + self.texture_weight + self.semantic_weight)
        
        # Обновляем счетчики для статистики
        self.total_count += fake_img.size(0)
        self.realistic_count += torch.sum((quality_score > 0.7).float()).item()
        
        return quality_score
    
    def get_statistics(self):
        """
        Возвращает текущие статистики качества.
        
        Returns:
            dict: Статистики качества
        """
        realism_rate = (self.realistic_count / max(self.total_count, 1)).item()
        
        avg_color_error = (self.color_errors / max(self.total_count, 1)).item()
        avg_texture_error = (self.texture_errors / max(self.total_count, 1)).item()
        avg_semantic_error = (self.semantic_errors / max(self.total_count, 1)).item()
        
        reward_stats = self.gan_loss.get_reward_stats()
        
        return {
            'realism_rate': realism_rate,
            'avg_color_error': avg_color_error,
            'avg_texture_error': avg_texture_error,
            'avg_semantic_error': avg_semantic_error,
            'reward_multiplier': reward_stats['reward_multiplier']
        }
    
    def reset_statistics(self):
        """
        Сбрасывает счетчики статистики.
        """
        self.realistic_count.zero_()
        self.total_count.zero_()
        self.color_errors.zero_()
        self.texture_errors.zero_()
        self.semantic_errors.zero_()
    
    def forward(self, discriminator, fake_img, real_img, fake_features=None, real_features=None, is_generator=True):
        """
        Вычисляет GAN-функцию потерь для задачи колоризации.
        
        Args:
            discriminator (nn.Module): Дискриминатор для оценки изображений
            fake_img (torch.Tensor): Сгенерированное изображение
            real_img (torch.Tensor): Реальное изображение
            fake_features (list, optional): Список сгенерированных признаков
            real_features (list, optional): Список реальных признаков
            is_generator (bool): Является ли это потерей для генератора
            
        Returns:
            dict: Словарь с результатами:
                - loss: Общая потеря
                - quality_score: Оценка качества изображения
                - gan_loss: Базовая GAN-потеря
                - color_loss: Потеря цветовых характеристик
        """
        # Вычисляем оценку качества изображения
        quality_score = self.compute_quality_score(fake_img, real_img, fake_features, real_features)
        
        # Разные потери в зависимости от режима работы
        if self.mode == 'global':
            # Глобальный дискриминатор оценивает изображение целиком
            if is_generator:
                # Генератор хочет обмануть дискриминатор
                pred_fake = discriminator(fake_img)
                gan_loss = self.gan_loss(pred_fake, True, quality_score, is_generator)
            else:
                # Дискриминатор учится различать реальные и поддельные изображения
                pred_real = discriminator(real_img)
                pred_fake = discriminator(fake_img.detach())
                
                gan_loss_real = self.gan_loss(pred_real, True, None, is_generator)
                gan_loss_fake = self.gan_loss(pred_fake, False, quality_score, is_generator)
                
                gan_loss = (gan_loss_real + gan_loss_fake) * 0.5
                
        elif self.mode == 'patchgan':
            # PatchGAN дискриминатор оценивает каждый патч изображения
            if is_generator:
                # Генератор хочет обмануть дискриминатор
                pred_fake = discriminator(fake_img)
                gan_loss = self.gan_loss(pred_fake, True, quality_score, is_generator)
            else:
                # Дискриминатор учится различать реальные и поддельные патчи
                pred_real = discriminator(real_img)
                pred_fake = discriminator(fake_img.detach())
                
                gan_loss_real = self.gan_loss(pred_real, True, None, is_generator)
                gan_loss_fake = self.gan_loss(pred_fake, False, quality_score, is_generator)
                
                gan_loss = (gan_loss_real + gan_loss_fake) * 0.5
                
        elif self.mode == 'multiscale':
            # Multiscale дискриминатор оценивает изображение на разных масштабах
            if is_generator:
                # Генератор хочет обмануть все дискриминаторы
                gan_loss = 0
                
                for scale_idx, scale_disc in enumerate(discriminator):
                    # Масштабирование изображения для текущего масштаба
                    scale_factor = 1 / (2 ** scale_idx)
                    if scale_factor < 1:
                        curr_fake_img = F.interpolate(fake_img, scale_factor=scale_factor, mode='bilinear')
                    else:
                        curr_fake_img = fake_img
                        
                    pred_fake = scale_disc(curr_fake_img)
                    curr_gan_loss = self.gan_loss(pred_fake, True, quality_score, is_generator)
                    gan_loss += curr_gan_loss
                
                # Усредняем потери по всем масштабам
                gan_loss = gan_loss / len(discriminator)
            else:
                # Дискриминаторы учатся различать реальные и поддельные изображения на разных масштабах
                gan_loss = 0
                
                for scale_idx, scale_disc in enumerate(discriminator):
                    # Масштабирование изображения для текущего масштаба
                    scale_factor = 1 / (2 ** scale_idx)
                    if scale_factor < 1:
                        curr_real_img = F.interpolate(real_img, scale_factor=scale_factor, mode='bilinear')
                        curr_fake_img = F.interpolate(fake_img.detach(), scale_factor=scale_factor, mode='bilinear')
                    else:
                        curr_real_img = real_img
                        curr_fake_img = fake_img.detach()
                        
                    pred_real = scale_disc(curr_real_img)
                    pred_fake = scale_disc(curr_fake_img)
                    
                    gan_loss_real = self.gan_loss(pred_real, True, None, is_generator)
                    gan_loss_fake = self.gan_loss(pred_fake, False, quality_score, is_generator)
                    
                    curr_gan_loss = (gan_loss_real + gan_loss_fake) * 0.5
                    gan_loss += curr_gan_loss
                
                # Усредняем потери по всем масштабам
                gan_loss = gan_loss / len(discriminator)
        else:
            raise ValueError(f"Неподдерживаемый режим: {self.mode}")
        
        # Дополнительные потери для генератора
        color_loss = 0
        if is_generator:
            # Извлекаем цветовые характеристики
            fake_color = self.extract_color_features(fake_img)
            real_color = self.extract_color_features(real_img)
            
            # Вычисляем потерю цвета
            color_loss = self.color_loss(fake_color, real_color)
            
            # Обновляем статистики генератора
            self.gan_loss.update_reward_multiplier(gan_loss.item(), 0)
            
        # Обновляем статистики дискриминатора
        if not is_generator:
            self.gan_loss.update_reward_multiplier(0, gan_loss.item())
        
        # Общая потеря
        loss = gan_loss + (0.5 * color_loss if is_generator else 0)
        
        return {
            'loss': loss,
            'quality_score': quality_score.detach().mean(),
            'gan_loss': gan_loss.detach(),
            'color_loss': color_loss if isinstance(color_loss, torch.Tensor) else torch.tensor(color_loss)
        }


class RewardPenaltyManager(nn.Module):
    """
    Менеджер системы наград и наказаний для колоризатора.
    
    Args:
        reward_scale (float): Масштаб наград
        penalty_scale (float): Масштаб наказаний
        use_adaptive_scaling (bool): Использовать ли адаптивное масштабирование
        max_reward (float): Максимальная награда
        max_penalty (float): Максимальное наказание
    """
    def __init__(self, reward_scale=1.0, penalty_scale=1.0, 
                 use_adaptive_scaling=True, max_reward=3.0, max_penalty=2.0):
        super(RewardPenaltyManager, self).__init__()
        
        self.reward_scale = reward_scale
        self.penalty_scale = penalty_scale
        self.use_adaptive_scaling = use_adaptive_scaling
        self.max_reward = max_reward
        self.max_penalty = max_penalty
        
        # История для адаптивного масштабирования
        self.register_buffer('reward_history', torch.zeros(100))
        self.register_buffer('penalty_history', torch.zeros(100))
        self.register_buffer('history_index', torch.tensor(0))
        
        # Счетчики для статистики
        self.register_buffer('total_rewards', torch.tensor(0.0))
        self.register_buffer('total_penalties', torch.tensor(0.0))
        self.register_buffer('reward_count', torch.tensor(0))
        self.register_buffer('penalty_count', torch.tensor(0))
        
    def compute_reward(self, quality_score, base_loss):
        """
        Вычисляет награду на основе качества и базовой потери.
        
        Args:
            quality_score (torch.Tensor): Оценка качества [0, 1]
            base_loss (torch.Tensor): Базовая функция потерь
            
        Returns:
            torch.Tensor: Модифицированная функция потерь с наградой
        """
        # Масштаб награды зависит от качества
        reward_factor = self.reward_scale * quality_score
        
        # Ограничиваем максимальную награду
        reward_factor = torch.clamp(reward_factor, 0, self.max_reward)
        
        # Обновляем статистику
        self.total_rewards += reward_factor.sum().item()
        self.reward_count += quality_score.numel()
        
        # Обновляем историю для адаптивного масштабирования
        if self.use_adaptive_scaling:
            idx = self.history_index % 100
            self.reward_history[idx] = reward_factor.mean().item()
            
        # Применяем награду к базовой потере
        # Меньшая потеря для высокого качества (награда)
        modified_loss = base_loss / (1 + reward_factor)
        
        return modified_loss
    
    def compute_penalty(self, quality_score, base_loss):
        """
        Вычисляет наказание на основе качества и базовой потери.
        
        Args:
            quality_score (torch.Tensor): Оценка качества [0, 1]
            base_loss (torch.Tensor): Базовая функция потерь
            
        Returns:
            torch.Tensor: Модифицированная функция потерь с наказанием
        """
        # Масштаб наказания обратно пропорционален качеству
        penalty_factor = self.penalty_scale * (1 - quality_score)
        
        # Ограничиваем максимальное наказание
        penalty_factor = torch.clamp(penalty_factor, 0, self.max_penalty)
        
        # Обновляем статистику
        self.total_penalties += penalty_factor.sum().item()
        self.penalty_count += quality_score.numel()
        
        # Обновляем историю для адаптивного масштабирования
        if self.use_adaptive_scaling:
            idx = self.history_index % 100
            self.penalty_history[idx] = penalty_factor.mean().item()
            self.history_index += 1
            
        # Применяем наказание к базовой потере
        # Большая потеря для низкого качества (наказание)
        modified_loss = base_loss * (1 + penalty_factor)
        
        return modified_loss
    
    def update_scaling_factors(self):
        """
        Обновляет масштабирующие факторы на основе истории.
        """
        if not self.use_adaptive_scaling or self.history_index < 50:
            return
            
        # Вычисляем средние значения наград и наказаний
        avg_reward = self.reward_history[:min(self.history_index.item(), 100)].mean().item()
        avg_penalty = self.penalty_history[:min(self.history_index.item(), 100)].mean().item()
        
        # Если награды слишком большие, уменьшаем масштаб
        if avg_reward > 1.5:
            self.reward_scale *= 0.95
        # Если награды слишком малые, увеличиваем масштаб
        elif avg_reward < 0.5:
            self.reward_scale *= 1.05
            
        # То же для наказаний
        if avg_penalty > 1.5:
            self.penalty_scale *= 0.95
        elif avg_penalty < 0.5:
            self.penalty_scale *= 1.05
            
        # Ограничиваем масштабы
        self.reward_scale = max(0.1, min(2.0, self.reward_scale))
        self.penalty_scale = max(0.1, min(2.0, self.penalty_scale))
    
    def get_statistics(self):
        """
        Возвращает текущие статистики системы наград и наказаний.
        
        Returns:
            dict: Статистики системы наград и наказаний
        """
        avg_reward = self.total_rewards / max(self.reward_count, 1)
        avg_penalty = self.total_penalties / max(self.penalty_count, 1)
        
        return {
            'avg_reward': avg_reward.item(),
            'avg_penalty': avg_penalty.item(),
            'reward_scale': self.reward_scale,
            'penalty_scale': self.penalty_scale
        }
    
    def reset_statistics(self):
        """
        Сбрасывает счетчики статистики.
        """
        self.total_rewards.zero_()
        self.total_penalties.zero_()
        self.reward_count.zero_()
        self.penalty_count.zero_()


class MotivationalColorLoss(nn.Module):
    """
    Мотивационная функция потерь для колоризации с системой наград и наказаний.
    
    Args:
        pixel_weight (float): Вес для попиксельной потери
        gan_weight (float): Вес для GAN потери
        color_consistency_weight (float): Вес для потери цветовой согласованности
        reward_scale (float): Масштаб наград
        penalty_scale (float): Масштаб наказаний
        gan_mode (str): Режим работы GAN ('vanilla', 'lsgan', 'wgan', 'hinge')
        use_adaptive_rewards (bool): Использовать ли адаптивные награды
    """
    def __init__(self, pixel_weight=10.0, gan_weight=1.0, color_consistency_weight=5.0,
                 reward_scale=1.0, penalty_scale=1.0, gan_mode='lsgan', use_adaptive_rewards=True):
        super(MotivationalColorLoss, self).__init__()
        
        # Веса для разных компонентов потери
        self.pixel_weight = pixel_weight
        self.gan_weight = gan_weight
        self.color_consistency_weight = color_consistency_weight
        
        # Базовые функции потерь
        self.pixel_loss = nn.L1Loss()
        self.color_consistency_loss = nn.SmoothL1Loss()
        
        # GAN функция потерь
        self.adversarial_loss = AdversarialColorLoss(
            mode='patchgan',
            color_space='lab',
            gan_mode=gan_mode
        )
        
        # Менеджер системы наград и наказаний
        self.reward_manager = RewardPenaltyManager(
            reward_scale=reward_scale,
            penalty_scale=penalty_scale,
            use_adaptive_scaling=use_adaptive_rewards
        )
        
        # Для отслеживания успехов и неудач
        self.register_buffer('success_counter', torch.tensor(0))
        self.register_buffer('failure_counter', torch.tensor(0))
        self.register_buffer('total_samples', torch.tensor(0))
    
    def compute_color_consistency(self, fake_img, real_img):
        """
        Вычисляет потерю цветовой согласованности между предсказанным и реальным изображениями.
        
        Args:
            fake_img (torch.Tensor): Сгенерированное изображение
            real_img (torch.Tensor): Реальное изображение
            
        Returns:
            torch.Tensor: Значение потери цветовой согласованности
        """
        # Извлекаем цветовые каналы
        if fake_img.size(1) >= 3:
            fake_color = fake_img[:, 1:3] if fake_img.size(1) == 3 else fake_img[:, 1:]
            real_color = real_img[:, 1:3] if real_img.size(1) == 3 else real_img[:, 1:]
        else:
            # Если меньше 3 каналов, используем все доступные
            fake_color = fake_img
            real_color = real_img
            
        # Вычисляем цветовые статистики для каждого изображения
        # Среднее значение цвета
        fake_mean = torch.mean(fake_color, dim=[2, 3], keepdim=True)
        real_mean = torch.mean(real_color, dim=[2, 3], keepdim=True)
        
        # Стандартное отклонение цвета
        fake_std = torch.std(fake_color, dim=[2, 3], keepdim=True)
        real_std = torch.std(real_color, dim=[2, 3], keepdim=True)
        
        # Потеря по средним значениям и стандартным отклонениям
        mean_loss = self.color_consistency_loss(fake_mean, real_mean)
        std_loss = self.color_consistency_loss(fake_std, real_std)
        
        # Потеря на гистограмме цветов (упрощенная версия)
        # Здесь мы просто используем сортированные значения цветов как приближение гистограммы
        fake_sorted = torch.sort(fake_color.reshape(fake_color.size(0), fake_color.size(1), -1), dim=2)[0]
        real_sorted = torch.sort(real_color.reshape(real_color.size(0), real_color.size(1), -1), dim=2)[0]
        
        # Берем выборку из отсортированных значений для эффективности
        sample_size = min(1000, fake_sorted.size(2))
        indices = torch.linspace(0, fake_sorted.size(2) - 1, sample_size).long()
        fake_sampled = fake_sorted[:, :, indices]
        real_sampled = real_sorted[:, :, indices]
        
        hist_loss = self.color_consistency_loss(fake_sampled, real_sampled)
        
        # Объединяем потери
        consistency_loss = mean_loss + std_loss + hist_loss
        
        return consistency_loss
    
    def evaluate_success(self, quality_score):
        """
        Оценивает успех или неудачу колоризации на основе оценки качества.
        
        Args:
            quality_score (torch.Tensor): Оценка качества колоризации [0, 1]
        """
        # Пороговые значения для успеха и неудачи
        success_threshold = 0.8
        failure_threshold = 0.3
        
        # Подсчет успехов и неудач
        successes = torch.sum((quality_score >= success_threshold).float()).item()
        failures = torch.sum((quality_score <= failure_threshold).float()).item()
        
        # Обновляем счетчики
        self.success_counter += successes
        self.failure_counter += failures
        self.total_samples += quality_score.numel()
    
    def get_motivation_stats(self):
        """
        Возвращает статистику мотивационной системы.
        
        Returns:
            dict: Статистика мотивационной системы
        """
        # Вычисляем процентные соотношения
        success_rate = (self.success_counter / max(self.total_samples, 1)).item()
        failure_rate = (self.failure_counter / max(self.total_samples, 1)).item()
        
        # Получаем статистику наград и наказаний
        reward_stats = self.reward_manager.get_statistics()
        
        # Получаем статистику GAN
        gan_stats = self.adversarial_loss.get_statistics()
        
        return {
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'reward_stats': reward_stats,
            'gan_stats': gan_stats
        }
    
    def reset_stats(self):
        """
        Сбрасывает все статистики.
        """
        self.success_counter.zero_()
        self.failure_counter.zero_()
        self.total_samples.zero_()
        self.reward_manager.reset_statistics()
        self.adversarial_loss.reset_statistics()
    
    def forward(self, discriminator, fake_img, real_img, gray_img=None, 
                fake_features=None, real_features=None, is_generator=True):
        """
        Вычисляет мотивационную функцию потерь для колоризации.
        
        Args:
            discriminator (nn.Module): Дискриминатор для оценки изображений
            fake_img (torch.Tensor): Сгенерированное изображение [B, C, H, W]
            real_img (torch.Tensor): Реальное изображение [B, C, H, W]
            gray_img (torch.Tensor, optional): Исходное черно-белое изображение [B, 1, H, W]
            fake_features (list, optional): Список признаков из генератора
            real_features (list, optional): Список признаков из эталонного изображения
            is_generator (bool): Является ли это потерей для генератора
            
        Returns:
            dict: Результаты с различными компонентами потерь и статистиками
        """
        # Для генератора вычисляем все компоненты потерь
        if is_generator:
            # Попиксельная потеря (L1)
            pixel_loss = self.pixel_loss(fake_img, real_img)
            
            # Потеря цветовой согласованности
            color_consistency_loss = self.compute_color_consistency(fake_img, real_img)
            
            # GAN-потеря
            gan_result = self.adversarial_loss(
                discriminator, fake_img, real_img, 
                fake_features, real_features, is_generator
            )
            
            gan_loss = gan_result['loss']
            quality_score = gan_result['quality_score']
            
            # Оцениваем успех или неудачу колоризации
            self.evaluate_success(quality_score.detach())
            
            # Базовая потеря без наград/наказаний
            base_loss = (
                self.pixel_weight * pixel_loss +
                self.gan_weight * gan_loss +
                self.color_consistency_weight * color_consistency_loss
            )
            
            # Применяем систему наград на основе качества
            if quality_score > 0.7:
                # Высокое качество - даем награду (уменьшаем потерю)
                final_loss = self.reward_manager.compute_reward(quality_score, base_loss)
            elif quality_score < 0.3:
                # Низкое качество - наказываем (увеличиваем потерю)
                final_loss = self.reward_manager.compute_penalty(quality_score, base_loss)
            else:
                # Среднее качество - без модификаций
                final_loss = base_loss
                
            # Обновляем масштабирующие факторы
            self.reward_manager.update_scaling_factors()
            
            return {
                'loss': final_loss,
                'pixel_loss': pixel_loss,
                'gan_loss': gan_loss,
                'color_consistency_loss': color_consistency_loss,
                'quality_score': quality_score,
                'base_loss': base_loss
            }
            
        else:  # Для дискриминатора только GAN-потеря
            gan_result = self.adversarial_loss(
                discriminator, fake_img, real_img, 
                fake_features, real_features, is_generator
            )
            
            return {
                'loss': gan_result['loss'],
                'gan_loss': gan_result['loss'],
                'quality_score': gan_result['quality_score']
            }


# Функция для создания GAN-функции потерь с заданными параметрами
def create_gan_loss(config=None):
    """
    Создает функцию потерь GAN для задачи колоризации с заданными параметрами.
    
    Args:
        config (dict, optional): Словарь параметров для функции потерь
        
    Returns:
        nn.Module: Функция потерь GAN
    """
    # Параметры по умолчанию
    default_config = {
        'gan_mode': 'lsgan',
        'pixel_weight': 10.0,
        'gan_weight': 1.0,
        'color_consistency_weight': 5.0,
        'reward_scale': 1.0,
        'penalty_scale': 1.0,
        'use_adaptive_rewards': True
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем функцию потерь
    loss_fn = MotivationalColorLoss(
        pixel_weight=default_config['pixel_weight'],
        gan_weight=default_config['gan_weight'],
        color_consistency_weight=default_config['color_consistency_weight'],
        reward_scale=default_config['reward_scale'],
        penalty_scale=default_config['penalty_scale'],
        gan_mode=default_config['gan_mode'],
        use_adaptive_rewards=default_config['use_adaptive_rewards']
    )
    
    return loss_fn


if __name__ == "__main__":
    # Пример использования
    
    # Создаем модель дискриминатора для тестирования
    class MockDiscriminator(nn.Module):
        def __init__(self):
            super(MockDiscriminator, self).__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=4, stride=2, padding=1)
            
        def forward(self, x):
            return self.conv(x)
    
    # Создаем функцию потерь
    loss_fn = create_gan_loss()
    
    # Создаем mock-дискриминатор
    discriminator = MockDiscriminator()
    
    # Создаем тестовые данные
    batch_size = 2
    fake_img = torch.rand(batch_size, 3, 32, 32)
    real_img = torch.rand(batch_size, 3, 32, 32)
    gray_img = torch.rand(batch_size, 1, 32, 32)
    
    # Вычисляем потери для генератора
    generator_result = loss_fn(discriminator, fake_img, real_img, gray_img)
    
    # Вычисляем потери для дискриминатора
    discriminator_result = loss_fn(discriminator, fake_img, real_img, gray_img, is_generator=False)
    
    # Выводим результаты
    print("Generator Loss:")
    for key, value in generator_result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item()}")
        else:
            print(f"  {key}: {value}")
            
    print("\nDiscriminator Loss:")
    for key, value in discriminator_result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item()}")
        else:
            print(f"  {key}: {value}")
            
    print("\nMotivation Stats:")
    motivation_stats = loss_fn.get_motivation_stats()
    for key, value in motivation_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")