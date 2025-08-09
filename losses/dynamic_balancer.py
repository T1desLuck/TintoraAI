"""
Dynamic Loss Balancing: Адаптивное взвешивание функций потерь для улучшения обучения.

Данный модуль реализует механизм динамического балансирования весов различных функций
потерь в процессе обучения колоризатора. Это позволяет модели адаптивно регулировать
важность разных компонентов потери в зависимости от их текущей динамики, сложности
обучения, и общего прогресса.

Ключевые особенности:
- Автоматическое определение "сложных" компонентов потери, требующих большего внимания
- Динамическая коррекция весов для сбалансированного обучения
- Механизмы анти-доминирования для предотвращения перевеса отдельных функций
- Отслеживание трендов изменения потерь для умной адаптации весов
- Возможность задания стратегий взвешивания с учетом этапов обучения

Преимущества для колоризации:
- Более стабильный процесс обучения
- Автоматическая адаптация к сложным образцам
- Предотвращение "забывания" важных аспектов при оптимизации других
- Лучший баланс между различными аспектами качества колоризации
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque


class LossHistoryTracker(nn.Module):
    """
    Отслеживание истории значений функций потерь для анализа трендов.
    
    Args:
        window_size (int): Размер окна для отслеживания истории
        num_losses (int): Количество отслеживаемых функций потерь
        device (torch.device): Устройство для тензоров
    """
    def __init__(self, window_size=100, num_losses=3, device=None):
        super(LossHistoryTracker, self).__init__()
        
        self.window_size = window_size
        self.num_losses = num_losses
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализируем историю потерь
        self.register_buffer('loss_history', torch.zeros(window_size, num_losses))
        self.register_buffer('history_index', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history_filled', torch.zeros(1, dtype=torch.bool))
        
        # Для вычисления скользящих средних и других статистик
        self.moving_avg_short = 10  # Короткое окно (последние 10 итераций)
        self.moving_avg_long = 50   # Длинное окно (последние 50 итераций)
        
        # Регистрируем средние значения как буферы
        self.register_buffer('moving_avg_short_values', torch.zeros(num_losses))
        self.register_buffer('moving_avg_long_values', torch.zeros(num_losses))
        self.register_buffer('initial_loss_values', torch.zeros(num_losses))
        self.register_buffer('initial_filled', torch.zeros(1, dtype=torch.bool))
        
    def update(self, losses):
        """
        Обновляет историю потерь новыми значениями.
        
        Args:
            losses (list or torch.Tensor): Список или тензор значений функций потерь
        """
        # Преобразуем в тензор, если необходимо
        if isinstance(losses, list):
            losses_tensor = torch.tensor(losses, device=self.device)
        else:
            losses_tensor = losses.detach().to(self.device)
            
        # Проверяем размерность
        assert losses_tensor.numel() == self.num_losses, \
            f"Ожидается {self.num_losses} значений потерь, получено {losses_tensor.numel()}"
        
        # Сохраняем первые значения потерь
        if not self.initial_filled[0]:
            self.initial_loss_values.copy_(losses_tensor)
            self.initial_filled[0] = True
            
        # Получаем текущий индекс в буфере истории
        idx = self.history_index[0] % self.window_size
        
        # Обновляем историю
        self.loss_history[idx].copy_(losses_tensor)
        
        # Увеличиваем индекс
        self.history_index[0] += 1
        
        # Отмечаем, что история заполнена, если достигнут размер окна
        if self.history_index[0] >= self.window_size:
            self.history_filled[0] = True
            
        # Обновляем скользящие средние
        self._update_moving_averages()
        
    def _update_moving_averages(self):
        """
        Обновляет скользящие средние значения потерь.
        """
        # Определяем индексы для вычисления средних
        history_len = min(self.history_index[0], self.window_size)
        
        if history_len > 0:
            # Короткое окно
            short_window = min(self.moving_avg_short, history_len)
            start_idx = (self.history_index[0] - short_window) % self.window_size
            
            # Если история еще не заполнила буфер полностью
            if not self.history_filled[0]:
                short_indices = torch.arange(short_window)
            else:
                # Вычисляем правильные индексы с учетом циклического буфера
                if start_idx + short_window <= self.window_size:
                    short_indices = torch.arange(start_idx, start_idx + short_window)
                else:
                    # Если окно пересекает конец буфера
                    first_part = torch.arange(start_idx, self.window_size)
                    second_part = torch.arange(0, short_window - (self.window_size - start_idx))
                    short_indices = torch.cat([first_part, second_part])
            
            # Вычисляем среднее для короткого окна
            self.moving_avg_short_values = torch.mean(self.loss_history[short_indices], dim=0)
            
            # Длинное окно
            if history_len >= self.moving_avg_long:
                long_window = self.moving_avg_long
                start_idx = (self.history_index[0] - long_window) % self.window_size
                
                # Вычисляем правильные индексы с учетом циклического буфера
                if start_idx + long_window <= self.window_size:
                    long_indices = torch.arange(start_idx, start_idx + long_window)
                else:
                    # Если окно пересекает конец буфера
                    first_part = torch.arange(start_idx, self.window_size)
                    second_part = torch.arange(0, long_window - (self.window_size - start_idx))
                    long_indices = torch.cat([first_part, second_part])
                
                # Вычисляем среднее для длинного окна
                self.moving_avg_long_values = torch.mean(self.loss_history[long_indices], dim=0)
            else:
                # Если история недостаточно длинная, используем все доступные значения
                if not self.history_filled[0]:
                    all_indices = torch.arange(history_len)
                else:
                    all_indices = torch.arange(self.window_size)
                
                self.moving_avg_long_values = torch.mean(self.loss_history[all_indices], dim=0)
    
    def get_trend_analysis(self):
        """
        Анализирует тренды изменения функций потерь.
        
        Returns:
            dict: Словарь с результатами анализа трендов
        """
        # Инициализируем результаты
        result = {
            'trends': [],             # Направление тренда (растет/падает/стабильно)
            'trend_strengths': [],    # Сила тренда
            'relative_to_initial': [] # Отношение к начальным значениям
        }
        
        # Если история недостаточно длинная, возвращаем нейтральные значения
        if self.history_index[0] < self.moving_avg_short:
            for _ in range(self.num_losses):
                result['trends'].append(0)  # Нейтральный тренд
                result['trend_strengths'].append(0.0)  # Нулевая сила тренда
                result['relative_to_initial'].append(1.0)  # Равно начальному значению
            return result
            
        # Анализируем тренды для каждой функции потерь
        for i in range(self.num_losses):
            # Короткое и длинное среднее
            short_avg = self.moving_avg_short_values[i].item()
            long_avg = self.moving_avg_long_values[i].item()
            
            # Отношение короткого среднего к длинному
            ratio = short_avg / (long_avg + 1e-8)
            
            # Определяем направление тренда
            if ratio < 0.9:
                trend = -1  # Падает
            elif ratio > 1.1:
                trend = 1   # Растет
            else:
                trend = 0   # Стабильно
                
            # Сила тренда (отклонение от 1.0)
            trend_strength = abs(ratio - 1.0)
            
            # Отношение к начальным значениям
            if self.initial_filled[0]:
                initial_value = self.initial_loss_values[i].item()
                rel_to_initial = short_avg / (initial_value + 1e-8)
            else:
                rel_to_initial = 1.0
                
            # Добавляем результаты
            result['trends'].append(trend)
            result['trend_strengths'].append(trend_strength)
            result['relative_to_initial'].append(rel_to_initial)
            
        return result


class DynamicWeightBalancer(nn.Module):
    """
    Динамическое балансирование весов функций потерь на основе их динамики.
    
    Args:
        loss_names (list): Список имен функций потерь
        initial_weights (list): Начальные веса функций потерь
        strategy (str): Стратегия балансировки ('dynamic', 'fixed', 'homoscedastic', 'gradnorm')
        learning_rate (float): Скорость обучения для весов
        use_softmax (bool): Использовать ли softmax для нормализации весов
        device (torch.device): Устройство для тензоров
    """
    def __init__(self, loss_names, initial_weights=None, strategy='dynamic',
                 learning_rate=0.01, use_softmax=True, device=None):
        super(DynamicWeightBalancer, self).__init__()
        
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.use_softmax = use_softmax
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализируем веса
        if initial_weights is None:
            initial_weights = [1.0] * self.num_losses
            
        assert len(initial_weights) == self.num_losses, \
            f"Количество начальных весов ({len(initial_weights)}) не соответствует количеству функций потерь ({self.num_losses})"
            
        # Логарифмические веса для удобства оптимизации
        log_weights = torch.log(torch.tensor(initial_weights, dtype=torch.float32))
        
        # Регистрируем веса как параметры
        self.log_weights = nn.Parameter(log_weights)
        
        # Для homoscedastic стратегии нужны логарифмические дисперсии
        if strategy == 'homoscedastic':
            self.log_vars = nn.Parameter(torch.zeros(self.num_losses))
            
        # Счетчик итераций
        self.register_buffer('iteration', torch.zeros(1, dtype=torch.long))
        
        # Трекер истории потерь
        self.history_tracker = LossHistoryTracker(
            window_size=100,
            num_losses=self.num_losses,
            device=self.device
        )
        
        # Пороги для защиты от доминирования одной функции потерь
        self.max_weight_ratio = 10.0  # Максимальное отношение весов
        self.min_weight = 0.01        # Минимальный вес
        
    def get_weights(self):
        """
        Возвращает текущие веса функций потерь.
        
        Returns:
            torch.Tensor: Веса функций потерь
        """
        # Преобразуем логарифмические веса в обычные
        weights = torch.exp(self.log_weights)
        
        # Применяем softmax, если требуется
        if self.use_softmax:
            weights = F.softmax(weights, dim=0)
        else:
            # Нормализуем веса, чтобы сумма была равна числу потерь
            weights = weights * self.num_losses / weights.sum()
            
        # Применяем ограничения на веса
        if not self.use_softmax:
            # Минимальный вес
            weights = torch.clamp(weights, min=self.min_weight)
            
            # Ограничение на максимальное отношение весов
            max_weight = torch.max(weights)
            min_weight = torch.min(weights)
            if max_weight / (min_weight + 1e-8) > self.max_weight_ratio:
                # Корректируем веса, чтобы отношение не превышало порог
                correction_factor = max_weight / (min_weight * self.max_weight_ratio)
                weights = weights / correction_factor
                
            # Нормализуем снова
            weights = weights * self.num_losses / weights.sum()
            
        return weights
        
    def update_weights(self, losses, gradients=None):
        """
        Обновляет веса функций потерь на основе их значений и градиентов.
        
        Args:
            losses (list or torch.Tensor): Значения функций потерь
            gradients (list or torch.Tensor, optional): Градиенты функций потерь
        """
        # Преобразуем в тензоры, если необходимо
        if isinstance(losses, list):
            losses = torch.tensor(losses, device=self.device)
        else:
            losses = losses.detach()
            
        if gradients is not None and isinstance(gradients, list):
            gradients = [g.detach() if isinstance(g, torch.Tensor) else torch.tensor(g, device=self.device) for g in gradients]
            
        # Обновляем историю потерь
        self.history_tracker.update(losses)
        
        # Увеличиваем счетчик итераций
        self.iteration += 1
        
        # Если итераций недостаточно, не обновляем веса
        if self.iteration < 50:
            return
            
        # Обновляем веса в зависимости от стратегии
        if self.strategy == 'fixed':
            # Фиксированные веса, не обновляем
            pass
            
        elif self.strategy == 'dynamic':
            # Динамическое обновление на основе истории потерь
            self._update_weights_dynamic()
            
        elif self.strategy == 'homoscedastic':
            # Гомоскедастическая неопределенность
            self._update_weights_homoscedastic(losses)
            
        elif self.strategy == 'gradnorm':
            # GradNorm
            if gradients is not None:
                self._update_weights_gradnorm(losses, gradients)
            else:
                # Если градиенты не предоставлены, используем динамическую стратегию
                self._update_weights_dynamic()
                
    def _update_weights_dynamic(self):
        """
        Обновляет веса на основе динамики функций потерь.
        """
        # Получаем анализ трендов
        trend_analysis = self.history_tracker.get_trend_analysis()
        
        # Получаем текущие веса
        weights = torch.exp(self.log_weights)
        
        # Обновляем веса на основе трендов
        for i in range(self.num_losses):
            trend = trend_analysis['trends'][i]
            trend_strength = trend_analysis['trend_strengths'][i]
            rel_to_initial = trend_analysis['relative_to_initial'][i]
            
            # Коэффициент обновления
            update_factor = 1.0
            
            # Если потеря растет (тренд > 0), увеличиваем ее вес
            if trend > 0:
                # Сила обновления зависит от силы тренда и отношения к начальному значению
                update_strength = trend_strength * min(2.0, rel_to_initial)
                update_factor = 1.0 + update_strength * self.learning_rate
            # Если потеря падает (тренд < 0), уменьшаем ее вес
            elif trend < 0:
                # Сила обновления зависит от силы тренда и обратного отношения к начальному значению
                update_strength = trend_strength * min(2.0, 1.0 / max(rel_to_initial, 0.1))
                update_factor = 1.0 / (1.0 + update_strength * self.learning_rate)
            
            # Применяем обновление
            weights[i] = weights[i] * update_factor
            
        # Обновляем логарифмические веса
        with torch.no_grad():
            self.log_weights.copy_(torch.log(weights))
            
    def _update_weights_homoscedastic(self, losses):
        """
        Обновляет веса на основе гомоскедастической неопределенности.
        
        Args:
            losses (torch.Tensor): Значения функций потерь
        """
        # В гомоскедастической стратегии веса обратно пропорциональны 
        # экспоненте логарифма дисперсии
        # Это метод автоматического взвешивания из Bayesian Deep Learning
        
        # Вместо прямого обновления весов, мы оптимизируем логарифмические дисперсии
        # в процессе обучения
        pass
    
    def _update_weights_gradnorm(self, losses, gradients):
        """
        Обновляет веса на основе алгоритма GradNorm.
        
        Args:
            losses (torch.Tensor): Значения функций потерь
            gradients (list): Список градиентов функций потерь
        """
        # GradNorm нормализует градиенты разных потерь к общему масштабу
        # Здесь мы реализуем упрощенную версию алгоритма
        
        # Вычисляем нормы градиентов
        grad_norms = torch.stack([torch.norm(grad) for grad in gradients])
        
        # Вычисляем среднюю норму
        mean_norm = torch.mean(grad_norms)
        
        # Обновляем веса, чтобы уравнять нормы градиентов
        weights = torch.exp(self.log_weights)
        
        for i in range(self.num_losses):
            # Корректирующий коэффициент
            if grad_norms[i] > 1e-6:  # Избегаем деления на очень маленькие значения
                correction = mean_norm / grad_norms[i]
                # Плавное обновление
                weights[i] = weights[i] * (1.0 - self.learning_rate + self.learning_rate * correction)
                
        # Обновляем логарифмические веса
        with torch.no_grad():
            self.log_weights.copy_(torch.log(weights))
    
    def forward(self, losses_dict=None, total_loss=None):
        """
        Вычисляет взвешенную сумму функций потерь.
        
        Args:
            losses_dict (dict): Словарь с функциями потерь {name: loss}
            total_loss (torch.Tensor, optional): Предварительно вычисленная общая потеря
            
        Returns:
            dict: Словарь с результатами {
                'total_loss': Взвешенная сумма потерь,
                'weighted_losses': Словарь с взвешенными потерями,
                'weights': Словарь с весами потерь
            }
        """
        # Получаем текущие веса
        weights = self.get_weights()
        
        # Если предоставлен словарь потерь, вычисляем взвешенную сумму
        if losses_dict is not None:
            # Проверяем, что все ожидаемые потери присутствуют
            for name in self.loss_names:
                assert name in losses_dict, f"Потеря '{name}' отсутствует в словаре"
                
            # Собираем значения потерь в том же порядке, что и имена
            losses = [losses_dict[name] for name in self.loss_names]
            
            # Вычисляем взвешенные потери
            weighted_losses = {}
            total_loss = 0.0
            
            for i, name in enumerate(self.loss_names):
                weighted_loss = weights[i] * losses[i]
                weighted_losses[name] = weighted_loss
                total_loss = total_loss + weighted_loss
                
        # Создаем словарь с весами
        weights_dict = {name: weights[i].item() for i, name in enumerate(self.loss_names)}
        
        # Результат
        result = {
            'total_loss': total_loss,
            'weighted_losses': weighted_losses if losses_dict is not None else None,
            'weights': weights_dict
        }
        
        return result


class AdaptiveLossBalancer(nn.Module):
    """
    Продвинутый балансировщик потерь с адаптивными стратегиями и анализом чувствительности.
    
    Args:
        loss_names (list): Список имен функций потерь
        initial_weights (list): Начальные веса функций потерь
        scheduler_type (str): Тип планировщика весов ('constant', 'linear', 'cosine')
        total_epochs (int): Общее количество эпох обучения
        adaptive_strategy (str): Стратегия адаптации ('dynamic', 'uncertainty', 'gradient', 'hybrid')
        min_weight_ratio (float): Минимальное отношение весов
        use_loss_uncertainty (bool): Учитывать ли неопределенность потерь
        device (torch.device): Устройство для тензоров
    """
    def __init__(self, loss_names, initial_weights=None, scheduler_type='cosine',
                 total_epochs=200, adaptive_strategy='hybrid', min_weight_ratio=0.1,
                 use_loss_uncertainty=True, device=None):
        super(AdaptiveLossBalancer, self).__init__()
        
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.adaptive_strategy = adaptive_strategy
        self.min_weight_ratio = min_weight_ratio
        self.use_loss_uncertainty = use_loss_uncertainty
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализируем веса
        if initial_weights is None:
            initial_weights = [1.0] * self.num_losses
            
        assert len(initial_weights) == self.num_losses, \
            f"Количество начальных весов ({len(initial_weights)}) не соответствует количеству функций потерь ({self.num_losses})"
            
        # Базовые веса (не обучаемые параметры)
        self.register_buffer('base_weights', torch.tensor(initial_weights, dtype=torch.float32))
        
        # Логарифмические веса для адаптивной стратегии (обучаемые параметры)
        self.log_adaptive_weights = nn.Parameter(torch.zeros(self.num_losses))
        
        # Буферы для отслеживания статистик
        self.register_buffer('epoch', torch.zeros(1, dtype=torch.long))
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('loss_means', torch.zeros(self.num_losses))
        self.register_buffer('loss_stds', torch.ones(self.num_losses))
        
        # Для отслеживания истории потерь и их градиентов
        self.loss_history = [deque(maxlen=100) for _ in range(self.num_losses)]
        self.grad_history = [deque(maxlen=100) for _ in range(self.num_losses)]
        
        # Для отслеживания чувствительности модели к разным потерям
        self.register_buffer('sensitivity', torch.ones(self.num_losses))
        
        # Для homoscedastic uncertainty weighting
        if self.use_loss_uncertainty:
            self.log_vars = nn.Parameter(torch.zeros(self.num_losses))
            
    def set_epoch(self, epoch):
        """
        Устанавливает текущую эпоху для планировщика весов.
        
        Args:
            epoch (int): Номер текущей эпохи
        """
        self.epoch[0] = epoch
        
    def step(self):
        """
        Увеличивает счетчик шагов.
        """
        self.step[0] += 1
        
    def get_scheduled_weights(self):
        """
        Возвращает веса согласно выбранному планировщику.
        
        Returns:
            torch.Tensor: Запланированные веса
        """
        if self.scheduler_type == 'constant':
            # Постоянные веса
            return self.base_weights
            
        elif self.scheduler_type == 'linear':
            # Линейное изменение весов
            progress = min(self.epoch[0].item() / self.total_epochs, 1.0)
            
            # Примерно: на начальных эпохах больше внимания пиксельной и перцептивной потерям,
            # затем постепенное увеличение веса GAN потери
            # Это пример, в реальном применении нужно настроить для конкретных потерь
            if self.num_losses == 3:  # Пример для трех потерь
                weights = torch.zeros_like(self.base_weights)
                # Первая потеря: линейно уменьшается
                weights[0] = self.base_weights[0] * (1.0 - 0.5 * progress)
                # Вторая потеря: сначала растет, потом уменьшается
                weights[1] = self.base_weights[1] * (1.0 + 0.5 * progress * (1 - progress) * 4)
                # Третья потеря: линейно растет
                weights[2] = self.base_weights[2] * (1.0 + progress)
                return weights
            else:
                # Для другого числа потерь возвращаем базовые веса
                return self.base_weights
                
        elif self.scheduler_type == 'cosine':
            # Косинусоидальное изменение весов
            progress = min(self.epoch[0].item() / self.total_epochs, 1.0)
            
            # Фазы для разных потерь (от 0 до 2π)
            phases = torch.linspace(0, 2 * math.pi, self.num_losses + 1)[:-1]
            
            # Вычисляем веса по косинусоидальному закону
            weights = self.base_weights.clone()
            for i in range(self.num_losses):
                # Амплитуда колебания (50% от базового веса)
                amplitude = 0.5 * self.base_weights[i]
                # Косинусоидальное изменение с заданной фазой
                modulation = 1.0 + amplitude * torch.cos(progress * math.pi + phases[i])
                weights[i] = self.base_weights[i] * modulation
                
            return weights
        else:
            # По умолчанию возвращаем базовые веса
            return self.base_weights
            
    def get_adaptive_weights(self):
        """
        Вычисляет адаптивные веса на основе выбранной стратегии.
        
        Returns:
            torch.Tensor: Адаптивные веса
        """
        # Преобразуем логарифмические веса
        adaptive_weights = torch.exp(self.log_adaptive_weights)
        
        # Применяем softmax для нормализации
        adaptive_weights = F.softmax(adaptive_weights, dim=0) * self.num_losses
        
        return adaptive_weights
        
    def get_uncertainty_weights(self):
        """
        Вычисляет веса на основе неопределенности потерь (homoscedastic uncertainty weighting).
        
        Returns:
            torch.Tensor: Веса на основе неопределенности
        """
        if not self.use_loss_uncertainty:
            return torch.ones(self.num_losses, device=self.device)
            
        # Преобразуем log_vars
        precision = torch.exp(-self.log_vars)
        
        # Веса пропорциональны точности (обратно пропорциональны дисперсии)
        weights = precision / torch.sum(precision) * self.num_losses
        
        return weights
        
    def update_statistics(self, losses, gradients=None):
        """
        Обновляет статистики потерь и градиентов.
        
        Args:
            losses (list or torch.Tensor): Значения функций потерь
            gradients (list, optional): Градиенты функций потерь
        """
        # Преобразуем в списки
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy().tolist()
        
        # Добавляем значения потерь в историю
        for i, loss_value in enumerate(losses):
            self.loss_history[i].append(loss_value)
        
        # Добавляем градиенты в историю, если они предоставлены
        if gradients is not None:
            for i, grad in enumerate(gradients):
                if isinstance(grad, torch.Tensor):
                    # Вычисляем норму градиента
                    grad_norm = torch.norm(grad).item()
                else:
                    grad_norm = grad
                self.grad_history[i].append(grad_norm)
        
        # Обновляем статистики
        for i in range(self.num_losses):
            if len(self.loss_history[i]) > 0:
                self.loss_means[i] = torch.tensor(np.mean(self.loss_history[i]), device=self.device)
                if len(self.loss_history[i]) > 1:
                    self.loss_stds[i] = torch.tensor(np.std(self.loss_history[i]), device=self.device)
                    
        # Обновляем чувствительность на основе градиентов
        if gradients is not None and all(len(h) > 0 for h in self.grad_history):
            grad_means = torch.tensor([np.mean(h) for h in self.grad_history], device=self.device)
            
            # Нормализуем средние градиенты
            if torch.sum(grad_means) > 0:
                normalized_grads = grad_means / torch.sum(grad_means) * self.num_losses
                
                # Плавное обновление чувствительности
                self.sensitivity = 0.9 * self.sensitivity + 0.1 * normalized_grads
                
    def update_adaptive_weights(self):
        """
        Обновляет адаптивные веса на основе выбранной стратегии.
        """
        if self.adaptive_strategy == 'dynamic':
            # Динамическая стратегия на основе статистик потерь
            self._update_weights_dynamic()
            
        elif self.adaptive_strategy == 'uncertainty':
            # Обновление на основе неопределенности потерь
            self._update_weights_uncertainty()
            
        elif self.adaptive_strategy == 'gradient':
            # Обновление на основе градиентов
            self._update_weights_gradient()
            
        elif self.adaptive_strategy == 'hybrid':
            # Гибридная стратегия
            self._update_weights_hybrid()
            
    def _update_weights_dynamic(self):
        """
        Обновляет веса на основе динамики потерь.
        """
        # Если история недостаточно длинная, не обновляем
        if any(len(h) < 20 for h in self.loss_history):
            return
            
        # Вычисляем тренды для каждой потери
        trends = []
        
        for i in range(self.num_losses):
            history = list(self.loss_history[i])
            
            # Разделяем историю на последние и предыдущие значения
            recent = np.mean(history[-10:])
            previous = np.mean(history[-20:-10])
            
            # Вычисляем изменение (отрицательное - улучшение, положительное - ухудшение)
            if previous > 0:
                change = (recent - previous) / previous
            else:
                change = 0.0
                
            trends.append(change)
            
        # Преобразуем в тензор
        trends = torch.tensor(trends, device=self.device)
        
        # Получаем текущие логарифмические веса
        current_log_weights = self.log_adaptive_weights.detach()
        
        # Обновляем веса
        with torch.no_grad():
            # Увеличиваем вес для потерь с положительным трендом (ухудшение)
            # и уменьшаем для потерь с отрицательным трендом (улучшение)
            # Масштабирующий фактор для обновления
            scale = 0.1
            self.log_adaptive_weights.copy_(current_log_weights + scale * trends)
            
    def _update_weights_uncertainty(self):
        """
        Обновляет веса на основе неопределенности потерь.
        """
        # Если не используем неопределенность, не обновляем
        if not self.use_loss_uncertainty:
            return
            
        # Неопределенность моделируется через log_vars, которые обновляются
        # во время обратного распространения, поэтому здесь ничего не делаем
        pass
        
    def _update_weights_gradient(self):
        """
        Обновляет веса на основе градиентов потерь.
        """
        # Если история градиентов недостаточно длинная, не обновляем
        if any(len(h) < 10 for h in self.grad_history):
            return
            
        # Используем чувствительность (средние нормализованные градиенты)
        # для обновления весов
        with torch.no_grad():
            # Инвертируем чувствительность: более высокие веса для меньших градиентов
            inverse_sensitivity = 1.0 / (self.sensitivity + 1e-8)
            
            # Нормализуем
            normalized_weights = inverse_sensitivity / torch.sum(inverse_sensitivity) * self.num_losses
            
            # Обновляем логарифмические веса
            self.log_adaptive_weights.copy_(torch.log(normalized_weights))
            
    def _update_weights_hybrid(self):
        """
        Обновляет веса используя гибридную стратегию.
        """
        # Комбинируем динамический подход и подход на основе градиентов
        if any(len(h) < 20 for h in self.loss_history) or any(len(h) < 10 for h in self.grad_history):
            return
            
        # Вычисляем динамические веса на основе трендов
        trends = []
        
        for i in range(self.num_losses):
            history = list(self.loss_history[i])
            
            # Разделяем историю на последние и предыдущие значения
            recent = np.mean(history[-10:])
            previous = np.mean(history[-20:-10])
            
            # Вычисляем изменение (отрицательное - улучшение, положительное - ухудшение)
            if previous > 0:
                change = (recent - previous) / previous
            else:
                change = 0.0
                
            trends.append(change)
            
        # Преобразуем в тензор
        trends = torch.tensor(trends, device=self.device)
        
        # Веса на основе трендов
        trend_weights = torch.exp(trends)
        
        # Веса на основе градиентов (инвертированные)
        inverse_sensitivity = 1.0 / (self.sensitivity + 1e-8)
        
        # Комбинируем оба подхода
        combined_weights = trend_weights * inverse_sensitivity
        
        # Нормализуем
        normalized_weights = combined_weights / torch.sum(combined_weights) * self.num_losses
        
        # Обновляем логарифмические веса
        with torch.no_grad():
            # Плавное обновление
            current_weights = torch.exp(self.log_adaptive_weights)
            updated_weights = 0.9 * current_weights + 0.1 * normalized_weights
            self.log_adaptive_weights.copy_(torch.log(updated_weights))
            
    def get_final_weights(self):
        """
        Вычисляет окончательные веса, комбинируя запланированные и адаптивные.
        
        Returns:
            torch.Tensor: Окончательные веса
        """
        # Получаем запланированные веса
        scheduled_weights = self.get_scheduled_weights()
        
        # Получаем адаптивные веса
        adaptive_weights = self.get_adaptive_weights()
        
        # Получаем веса на основе неопределенности
        uncertainty_weights = self.get_uncertainty_weights()
        
        # Комбинируем все веса
        # Влияние адаптивных весов увеличивается со временем
        progress = min(self.epoch[0].item() / (self.total_epochs * 0.5), 1.0)
        adaptive_factor = 0.5 * progress
        
        # Итоговые веса
        final_weights = (
            (1.0 - adaptive_factor) * scheduled_weights +
            adaptive_factor * adaptive_weights
        ) * uncertainty_weights
        
        # Нормализуем
        final_weights = final_weights / torch.sum(final_weights) * self.num_losses
        
        # Применяем ограничения на минимальное отношение весов
        max_weight = torch.max(final_weights)
        min_weight = torch.min(final_weights)
        
        # Если отношение весов слишком большое, корректируем
        if min_weight > 0 and max_weight / min_weight > 1.0 / self.min_weight_ratio:
            # Вычисляем целевое минимальное значение
            target_min = max_weight * self.min_weight_ratio
            
            # Линейно масштабируем веса, чтобы минимальное значение было равно target_min
            if min_weight < target_min:
                scale = (target_min - min_weight) / (max_weight - min_weight)
                final_weights = min_weight + scale * (final_weights - min_weight)
                
                # Нормализуем снова
                final_weights = final_weights / torch.sum(final_weights) * self.num_losses
        
        return final_weights
    
    def forward(self, losses_dict, total_loss=None, update_stats=True, gradients=None):
        """
        Вычисляет взвешенную сумму функций потерь.
        
        Args:
            losses_dict (dict): Словарь с функциями потерь {name: loss}
            total_loss (torch.Tensor, optional): Предварительно вычисленная общая потеря
            update_stats (bool): Обновлять ли статистики
            gradients (dict, optional): Словарь с градиентами {name: gradient}
            
        Returns:
            dict: Словарь с результатами {
                'total_loss': Взвешенная сумма потерь,
                'weighted_losses': Словарь с взвешенными потерями,
                'weights': Словарь с весами потерь
            }
        """
        # Проверяем, что все ожидаемые потери присутствуют
        for name in self.loss_names:
            assert name in losses_dict, f"Потеря '{name}' отсутствует в словаре"
            
        # Собираем значения потерь в том же порядке, что и имена
        losses = [losses_dict[name] for name in self.loss_names]
        
        # Собираем градиенты, если они предоставлены
        grads = None
        if gradients is not None:
            grads = [gradients.get(name, None) for name in self.loss_names]
            
        # Обновляем статистики, если требуется
        if update_stats:
            self.update_statistics(losses, grads)
            self.update_adaptive_weights()
        
        # Получаем окончательные веса
        weights = self.get_final_weights()
        
        # Вычисляем взвешенные потери
        weighted_losses = {}
        
        if total_loss is None:
            total_loss = 0.0
            
            for i, name in enumerate(self.loss_names):
                # Применяем вес к потере
                weighted_loss = weights[i] * losses[i]
                weighted_losses[name] = weighted_loss
                total_loss = total_loss + weighted_loss
                
        # Если используется homoscedastic uncertainty, добавляем регуляризацию log_vars
        if self.use_loss_uncertainty:
            total_loss = total_loss + 0.5 * torch.sum(self.log_vars)
        
        # Создаем словарь с весами
        weights_dict = {name: weights[i].item() for i, name in enumerate(self.loss_names)}
        
        # Результат
        result = {
            'total_loss': total_loss,
            'weighted_losses': weighted_losses,
            'weights': weights_dict
        }
        
        return result


class DynamicLossBalancer(nn.Module):
    """
    Комплексный балансировщик потерь, объединяющий функциональность всех балансировщиков.
    
    Этот класс предоставляет единый интерфейс к системе балансировки потерь,
    используя комбинацию возможностей LossHistoryTracker, DynamicWeightBalancer
    и AdaptiveLossBalancer в зависимости от настроек и текущей фазы обучения.
    
    Args:
        loss_names (list): Список имен функций потерь
        initial_weights (list): Начальные веса функций потерь
        strategy (str): Стратегия балансировки ('adaptive', 'dynamic', 'fixed')
        scheduler_type (str): Тип планировщика весов для адаптивной стратегии
        total_epochs (int): Общее количество эпох обучения
        adaptive_strategy (str): Подстратегия для адаптивного режима
        min_weight_ratio (float): Минимальное отношение весов
        use_loss_uncertainty (bool): Использовать ли неопределенность потерь
        device (torch.device): Устройство для тензоров
    """
    def __init__(self, loss_names, initial_weights=None, strategy='adaptive',
                 scheduler_type='cosine', total_epochs=200, adaptive_strategy='hybrid',
                 min_weight_ratio=0.1, use_loss_uncertainty=True, device=None):
        super(DynamicLossBalancer, self).__init__()
        
        self.strategy = strategy
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        
        # Создаем трекер истории потерь для всех стратегий
        self.history_tracker = LossHistoryTracker(
            window_size=100,
            num_losses=self.num_losses,
            device=device
        )
        
        # В зависимости от стратегии инициализируем соответствующий балансировщик
        if strategy == 'adaptive':
            self.balancer = AdaptiveLossBalancer(
                loss_names=loss_names,
                initial_weights=initial_weights,
                scheduler_type=scheduler_type,
                total_epochs=total_epochs,
                adaptive_strategy=adaptive_strategy,
                min_weight_ratio=min_weight_ratio,
                use_loss_uncertainty=use_loss_uncertainty,
                device=device
            )
        else:  # 'dynamic' или 'fixed'
            self.balancer = DynamicWeightBalancer(
                loss_names=loss_names,
                initial_weights=initial_weights,
                strategy=strategy,
                device=device
            )
    
    def forward(self, losses, update_stats=True, gradients=None):
        """
        Вычисляет взвешенные потери с использованием выбранной стратегии балансировки.
        
        Args:
            losses: Словарь с потерями или список значений потерь
            update_stats (bool): Обновлять ли статистику потерь
            gradients: Градиенты потерь для градиентных стратегий
            
        Returns:
            dict: Результат с взвешенными потерями, общей потерей и текущими весами
        """
        # Обновляем историю потерь, если нужно
        if update_stats:
            if isinstance(losses, dict):
                loss_values = [losses[name] for name in self.loss_names]
                self.history_tracker.update(loss_values)
            else:
                self.history_tracker.update(losses)
        
        # Делегируем работу соответствующему балансировщику
        return self.balancer(losses, update_stats=update_stats, gradients=gradients)
    
    def set_epoch(self, epoch):
        """
        Устанавливает текущую эпоху для планировщика весов.
        
        Args:
            epoch (int): Номер текущей эпохи
        """
        if hasattr(self.balancer, 'set_epoch'):
            self.balancer.set_epoch(epoch)
    
    def get_weights(self):
        """
        Возвращает текущие веса для функций потерь.
        
        Returns:
            dict: Словарь {имя_потери: вес}
        """
        return self.balancer.get_weights() if hasattr(self.balancer, 'get_weights') else None
    
    def get_loss_trends(self):
        """
        Возвращает тренды изменения потерь.
        
        Returns:
            dict: Информация о трендах потерь
        """
        return self.history_tracker.get_trend_analysis()


# Функция для создания балансировщика потерь
def create_dynamic_loss_balancer(loss_names, config=None):
    """
    Создает Dynamic Loss Balancer с заданной конфигурацией.
    
    Args:
        loss_names (list): Список имен функций потерь
        config (dict, optional): Конфигурация балансировщика
        
    Returns:
        nn.Module: Балансировщик потерь
    """
    # Параметры по умолчанию
    default_config = {
        'strategy': 'adaptive',  # 'fixed', 'dynamic', 'adaptive'
        'initial_weights': None,
        'scheduler_type': 'cosine',
        'total_epochs': 200,
        'adaptive_strategy': 'hybrid',
        'min_weight_ratio': 0.1,
        'use_loss_uncertainty': True
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Выбираем класс балансировщика в зависимости от стратегии
    if default_config['strategy'] == 'fixed':
        balancer = DynamicWeightBalancer(
            loss_names=loss_names,
            initial_weights=default_config['initial_weights'],
            strategy='fixed'
        )
    elif default_config['strategy'] == 'dynamic':
        balancer = DynamicWeightBalancer(
            loss_names=loss_names,
            initial_weights=default_config['initial_weights'],
            strategy='dynamic',
            learning_rate=0.01,
            use_softmax=True
        )
    else:  # 'adaptive'
        balancer = AdaptiveLossBalancer(
            loss_names=loss_names,
            initial_weights=default_config['initial_weights'],
            scheduler_type=default_config['scheduler_type'],
            total_epochs=default_config['total_epochs'],
            adaptive_strategy=default_config['adaptive_strategy'],
            min_weight_ratio=default_config['min_weight_ratio'],
            use_loss_uncertainty=default_config['use_loss_uncertainty']
        )
    
    return balancer


if __name__ == "__main__":
    # Пример использования
    
    # Имена функций потерь
    loss_names = ['pixel_loss', 'perceptual_loss', 'gan_loss']
    
    # Создаем балансировщик
    balancer = create_dynamic_loss_balancer(loss_names, {
        'strategy': 'adaptive',
        'initial_weights': [10.0, 1.0, 0.5],
        'total_epochs': 100
    })
    
    # Эмулируем обучение
    for epoch in range(10):
        # Устанавливаем текущую эпоху
        balancer.set_epoch(epoch)
        
        # Эмулируем батчи
        for batch in range(5):
            # Генерируем случайные потери и градиенты для примера
            pixel_loss = torch.rand(1) * (10.0 - epoch * 0.5)  # Уменьшается со временем
            perceptual_loss = torch.rand(1) * 2.0
            gan_loss = torch.rand(1) * (0.5 + epoch * 0.1)  # Увеличивается со временем
            
            # Словарь потерь
            losses = {
                'pixel_loss': pixel_loss,
                'perceptual_loss': perceptual_loss,
                'gan_loss': torch.rand(1) * 2.0
            }
            
            # Вычисляем взвешенную сумму потерь
            result = balancer(losses, update_stats=True, gradients=gradients)
            
            # Выводим результаты
            if batch == 0:
                print(f"Epoch {epoch}")
                print(f"  Weights: {result['weights']}")
                print(f"  Total Loss: {result['total_loss'].item()}")
                print(f"  Weighted Losses: {[(k, v.item()) for k, v in result['weighted_losses'].items()]}")
                print("")
