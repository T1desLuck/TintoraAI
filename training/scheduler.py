"""
Scheduler: Модуль для управления скоростью обучения моделей колоризации.

Данный модуль предоставляет функциональность для создания и управления планировщиками 
скорости обучения (learning rate schedulers), адаптируя темп обучения в соответствии 
с ходом тренировки, чтобы достичь лучших результатов и избежать проблем с застреванием 
в локальных минимумах.

Ключевые особенности:
- Поддержка различных стратегий планирования скорости обучения
- Специализированные планировщики для моделей колоризации
- Возможность комбинирования нескольких стратегий
- Управление на основе эпох или шагов обучения
- Адаптация на основе метрик качества

Преимущества:
- Ускорение процесса сходимости модели
- Повышение качества колоризации
- Предотвращение переобучения и осцилляций
- Гибкая настройка через конфигурационные файлы
"""

import math
from typing import Dict, List, Union, Optional, Any, Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR, StepLR, MultiStepLR, ExponentialLR, 
    CosineAnnealingLR, ReduceLROnPlateau, CyclicLR,
    OneCycleLR, _LRScheduler
)


class GradualWarmupScheduler(_LRScheduler):
    """
    Планировщик с постепенным разогревом (warmup) и последующим переходом на основной планировщик.
    
    Args:
        optimizer (Optimizer): Оптимизатор
        warmup_epochs (int): Количество эпох разогрева
        after_scheduler (Optional[_LRScheduler]): Планировщик, который будет использоваться после разогрева
        init_lr_factor (float): Начальный множитель скорости обучения (относительно базовой lr)
        warmup_mode (str): Режим разогрева ('linear' или 'exponential')
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        after_scheduler: Optional[_LRScheduler] = None,
        init_lr_factor: float = 0.1,
        warmup_mode: str = 'linear'
    ):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.init_lr_factor = init_lr_factor
        self.warmup_mode = warmup_mode
        self.finished = False
        super().__init__(optimizer)
    
    def get_lr(self) -> List[float]:
        """
        Вычисляет текущие значения скорости обучения.
        
        Returns:
            List[float]: Список текущих значений скорости обучения
        """
        if self.last_epoch > self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return self.base_lrs
            
        if self.warmup_mode == 'linear':
            # Линейное увеличение от init_lr_factor * base_lr до base_lr
            alpha = self.last_epoch / self.warmup_epochs
            factor = self.init_lr_factor + (1 - self.init_lr_factor) * alpha
        else:  # 'exponential'
            # Экспоненциальное увеличение от init_lr_factor * base_lr до base_lr
            factor = self.init_lr_factor * (1 / self.init_lr_factor) ** (self.last_epoch / self.warmup_epochs)
            
        return [base_lr * factor for base_lr in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Обновляет состояние планировщика.
        
        Args:
            epoch (int, optional): Текущая эпоха
        """
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Планировщик с косинусным затуханием и периодическим перезапуском с разогревом.
    
    Args:
        optimizer (Optimizer): Оптимизатор
        T_0 (int): Количество эпох до первого перезапуска
        T_mult (int): Множитель для увеличения периода перезапуска
        eta_min (float): Минимальное значение скорости обучения
        warmup_epochs (int): Количество эпох разогрева при каждом перезапуске
        warmup_factor (float): Множитель скорости обучения в начале разогрева
    """
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        warmup_epochs: int = 0,
        warmup_factor: float = 0.1
    ):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.T_cur = 0
        self.cycle_count = 0
        super().__init__(optimizer)
    
    def get_lr(self) -> List[float]:
        """
        Вычисляет текущие значения скорости обучения.
        
        Returns:
            List[float]: Список текущих значений скорости обучения
        """
        # Определяем, находимся ли мы в периоде разогрева текущего цикла
        warmup_epoch = self.warmup_epochs * (self.cycle_count + 1)
        if self.last_epoch < warmup_epoch:
            # Линейный разогрев
            alpha = (self.last_epoch - self.warmup_epochs * self.cycle_count) / self.warmup_epochs
            factor = self.warmup_factor + (1 - self.warmup_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        
        # Косинусное затухание
        if self.T_cur == 0 and self.last_epoch > 0:
            # Начало нового цикла
            self.cycle_count += 1
            if self.T_mult != 1:
                self.T_i = self.T_i * self.T_mult
                
        # Обновляем T_cur
        if self.last_epoch >= warmup_epoch:
            self.T_cur = self.last_epoch - warmup_epoch
            
        # Косинусное затухание
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / (self.T_i - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Обновляет состояние планировщика.
        
        Args:
            epoch (int, optional): Текущая эпоха
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.last_epoch = epoch
        
        # Проверяем, нужно ли начать новый цикл
        warmup_epoch = self.warmup_epochs * (self.cycle_count + 1)
        if self.last_epoch >= warmup_epoch:
            if self.T_cur + 1 == self.T_i - self.warmup_epochs:
                self.T_cur = 0
            else:
                self.T_cur += 1
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
        self._last_lr = self.get_lr()


class AdaptiveLRScheduler(_LRScheduler):
    """
    Адаптивный планировщик, регулирующий скорость обучения на основе изменения метрики.
    
    Args:
        optimizer (Optimizer): Оптимизатор
        mode (str): Режим изменения метрики ('min' - меньше лучше, 'max' - больше лучше)
        patience (int): Количество эпох ожидания улучшения метрики
        factor (float): Множитель для уменьшения скорости обучения
        threshold (float): Порог изменения метрики для считывания улучшения
        cooldown (int): Количество эпох ожидания после изменения скорости обучения
        min_lr (float or List): Минимальное значение скорости обучения
        eps (float): Минимальное значимое изменение скорости обучения
    """
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = 'min',
        patience: int = 10,
        factor: float = 0.5,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8
    ):
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self.eps = eps
        
        # Преобразуем min_lr в список
        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
            
        super().__init__(optimizer)
    
    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = None) -> None:
        """
        Обновляет состояние планировщика на основе метрики.
        
        Args:
            metrics (float, optional): Значение метрики
            epoch (int, optional): Текущая эпоха
        """
        # Обновляем последнюю эпоху
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch = self.last_epoch + 1
            
        # Пропускаем обновление, если metrics не предоставлен
        if metrics is None:
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            return
            
        # Пропускаем период cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            
        # Определяем, улучшилась ли метрика
        if self.best is None or self._is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        # Уменьшаем скорость обучения, если метрика не улучшается достаточно долго
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
    def _is_better(self, current: float, best: float) -> bool:
        """
        Определяет, лучше ли текущее значение метрики, чем лучшее.
        
        Args:
            current (float): Текущее значение метрики
            best (float): Лучшее значение метрики
            
        Returns:
            bool: True, если текущее значение лучше
        """
        if self.mode == 'min':
            return current < best * (1 - self.threshold)
        else:  # 'max'
            return current > best * (1 + self.threshold)
            
    def _reduce_lr(self, epoch: int) -> None:
        """
        Уменьшает скорость обучения.
        
        Args:
            epoch (int): Текущая эпоха
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            
            # Проверяем, что изменение значимое
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                
                print(f'Epoch {epoch}: AdaptiveLRScheduler уменьшает скорость обучения группы {i} до {new_lr:.6f}')


def create_scheduler(
    optimizer: Optimizer,
    config: Dict
) -> Optional[_LRScheduler]:
    """
    Создает планировщик скорости обучения на основе конфигурации.
    
    Args:
        optimizer (Optimizer): Оптимизатор
        config (Dict): Конфигурация планировщика
        
    Returns:
        Optional[_LRScheduler]: Созданный планировщик или None
    """
    # Проверяем, нужно ли создавать планировщик
    scheduler_type = config.get('type', 'none')
    if scheduler_type == 'none':
        return None
        
    # Создаем соответствующий планировщик
    if scheduler_type == 'step':
        # Понижение скорости обучения через заданное количество эпох
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_type == 'multistep':
        # Понижение скорости обучения в заданных эпохах
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
    elif scheduler_type == 'exponential':
        # Экспоненциальное понижение скорости обучения
        gamma = config.get('gamma', 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        
    elif scheduler_type == 'cosine':
        # Косинусное затухание скорости обучения
        T_max = config.get('T_max', 100)
        eta_min = config.get('eta_min', 0)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
    elif scheduler_type == 'plateau':
        # Понижение скорости обучения при отсутствии улучшения метрики
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.1)
        patience = config.get('patience', 10)
        threshold = config.get('threshold', 1e-4)
        cooldown = config.get('cooldown', 0)
        min_lr = config.get('min_lr', 1e-6)
        eps = config.get('eps', 1e-8)
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, cooldown=cooldown, min_lr=min_lr, eps=eps
        )
        
    elif scheduler_type == 'cyclic':
        # Циклическое изменение скорости обучения
        base_lr = config.get('base_lr', 1e-5)
        max_lr = config.get('max_lr', 1e-2)
        step_size_up = config.get('step_size_up', 2000)
        mode = config.get('mode', 'triangular')
        
        scheduler = CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr,
            step_size_up=step_size_up, mode=mode
        )
        
    elif scheduler_type == 'one_cycle':
        # One Cycle Policy
        max_lr = config.get('max_lr', 1e-2)
        epochs = config.get('epochs', 100)
        steps_per_epoch = config.get('steps_per_epoch', 100)
        pct_start = config.get('pct_start', 0.3)
        anneal_strategy = config.get('anneal_strategy', 'cos')
        
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=pct_start, anneal_strategy=anneal_strategy
        )
        
    elif scheduler_type == 'warmup_cosine':
        # Разогрев с последующим косинусным затуханием
        warmup_epochs = config.get('warmup_epochs', 5)
        T_max = config.get('T_max', 100)
        eta_min = config.get('eta_min', 0)
        init_lr_factor = config.get('init_lr_factor', 0.1)
        
        after_scheduler = CosineAnnealingLR(optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min)
        
        scheduler = GradualWarmupScheduler(
            optimizer, warmup_epochs=warmup_epochs, after_scheduler=after_scheduler,
            init_lr_factor=init_lr_factor, warmup_mode='linear'
        )
        
    elif scheduler_type == 'warmup_multistep':
        # Разогрев с последующим пошаговым понижением
        warmup_epochs = config.get('warmup_epochs', 5)
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        init_lr_factor = config.get('init_lr_factor', 0.1)
        
        # Корректируем вехи с учетом периода разогрева
        adjusted_milestones = [m - warmup_epochs for m in milestones if m > warmup_epochs]
        
        after_scheduler = MultiStepLR(optimizer, milestones=adjusted_milestones, gamma=gamma)
        
        scheduler = GradualWarmupScheduler(
            optimizer, warmup_epochs=warmup_epochs, after_scheduler=after_scheduler,
            init_lr_factor=init_lr_factor, warmup_mode='linear'
        )
        
    elif scheduler_type == 'cosine_warm_restarts':
        # Косинусное затухание с перезапусками и разогревом
        T_0 = config.get('T_0', 10)
        T_mult = config.get('T_mult', 2)
        eta_min = config.get('eta_min', 1e-6)
        warmup_epochs = config.get('warmup_epochs', 2)
        warmup_factor = config.get('warmup_factor', 0.1)
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
            warmup_epochs=warmup_epochs, warmup_factor=warmup_factor
        )
        
    elif scheduler_type == 'adaptive':
        # Адаптивный планировщик
        mode = config.get('mode', 'min')
        patience = config.get('patience', 10)
        factor = config.get('factor', 0.5)
        threshold = config.get('threshold', 1e-4)
        cooldown = config.get('cooldown', 0)
        min_lr = config.get('min_lr', 1e-6)
        
        scheduler = AdaptiveLRScheduler(
            optimizer, mode=mode, patience=patience, factor=factor,
            threshold=threshold, cooldown=cooldown, min_lr=min_lr
        )
        
    else:
        raise ValueError(f"Неизвестный тип планировщика: {scheduler_type}")
        
    return scheduler


def create_lr_lambda(
    config: Dict
) -> Callable[[int], float]:
    """
    Создает функцию lambda для планировщика LambdaLR.
    
    Args:
        config (Dict): Конфигурация функции lambda
        
    Returns:
        Callable[[int], float]: Функция lambda для планировщика
    """
    lambda_type = config.get('type', 'constant')
    
    if lambda_type == 'constant':
        # Постоянная скорость обучения
        return lambda epoch: 1.0
        
    elif lambda_type == 'linear_decay':
        # Линейное уменьшение скорости обучения
        total_epochs = config.get('total_epochs', 100)
        end_factor = config.get('end_factor', 0.01)
        
        return lambda epoch: max(end_factor, 1.0 - (1.0 - end_factor) * epoch / total_epochs)
        
    elif lambda_type == 'cosine_decay':
        # Косинусное уменьшение скорости обучения
        total_epochs = config.get('total_epochs', 100)
        end_factor = config.get('end_factor', 0.01)
        
        return lambda epoch: end_factor + (1.0 - end_factor) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / total_epochs)
        )
        
    elif lambda_type == 'exponential_decay':
        # Экспоненциальное уменьшение скорости обучения
        gamma = config.get('gamma', 0.95)
        
        return lambda epoch: gamma ** epoch
        
    elif lambda_type == 'step_decay':
        # Пошаговое уменьшение скорости обучения
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        
        return lambda epoch: gamma ** (epoch // step_size)
        
    elif lambda_type == 'custom':
        # Пользовательская функция, определенная через конфигурацию
        points = config.get('points', [(0, 1.0), (100, 0.01)])
        interpolation = config.get('interpolation', 'linear')
        
        epochs = [p[0] for p in points]
        factors = [p[1] for p in points]
        
        def custom_lambda(epoch: int) -> float:
            if epoch <= epochs[0]:
                return factors[0]
                
            if epoch >= epochs[-1]:
                return factors[-1]
                
            # Находим соседние точки
            for i in range(len(epochs) - 1):
                if epochs[i] <= epoch < epochs[i + 1]:
                    start_epoch, end_epoch = epochs[i], epochs[i + 1]
                    start_factor, end_factor = factors[i], factors[i + 1]
                    
                    if interpolation == 'linear':
                        # Линейная интерполяция
                        alpha = (epoch - start_epoch) / (end_epoch - start_epoch)
                        return start_factor + alpha * (end_factor - start_factor)
                    elif interpolation == 'cosine':
                        # Косинусная интерполяция
                        alpha = 0.5 * (1 + math.cos(math.pi * (1 - (epoch - start_epoch) / (end_epoch - start_epoch))))
                        return start_factor * alpha + end_factor * (1 - alpha)
                    else:
                        # По умолчанию - линейная интерполяция
                        alpha = (epoch - start_epoch) / (end_epoch - start_epoch)
                        return start_factor + alpha * (end_factor - start_factor)
                    
            return factors[-1]  # В случае ошибки возвращаем последний фактор
            
        return custom_lambda
        
    else:
        raise ValueError(f"Неизвестный тип lambda-функции: {lambda_type}")


def create_lambda_scheduler(
    optimizer: Optimizer,
    config: Dict
) -> LambdaLR:
    """
    Создает планировщик LambdaLR на основе конфигурации.
    
    Args:
        optimizer (Optimizer): Оптимизатор
        config (Dict): Конфигурация планировщика
        
    Returns:
        LambdaLR: Созданный планировщик
    """
    lr_lambda = create_lr_lambda(config)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


if __name__ == "__main__":
    # Пример использования
    import torch.optim as optim
    import matplotlib.pyplot as plt
    
    # Создаем простую модель и оптимизатор
    model = torch.nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Проверяем различные планировщики
    schedulers = {
        "Step": create_scheduler(optimizer, {"type": "step", "step_size": 20, "gamma": 0.5}),
        "Cosine": create_scheduler(optimizer, {"type": "cosine", "T_max": 100, "eta_min": 1e-5}),
        "Warm Cosine": create_scheduler(optimizer, {"type": "warmup_cosine", "warmup_epochs": 10, "T_max": 100, "eta_min": 1e-5}),
        "Cosine Warm Restarts": create_scheduler(optimizer, {"type": "cosine_warm_restarts", "T_0": 20, "T_mult": 2, "eta_min": 1e-5})
    }
    
    # Визуализируем изменение скорости обучения
    epochs = 100
    plt.figure(figsize=(12, 6))
    
    for name, scheduler in schedulers.items():
        lrs = []
        
        # Сбрасываем скорость обучения
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
            
        # Собираем скорости обучения для каждой эпохи
        for epoch in range(epochs):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
            
        plt.plot(range(epochs), lrs, label=name)
        
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_schedulers.png')
    
    print("Готово! График сохранен в файл lr_schedulers.png")