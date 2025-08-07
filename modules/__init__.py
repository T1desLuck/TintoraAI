"""
TintoraAI Modules: Интеллектуальные модули для улучшения колоризации изображений.

Данный модуль объединяет набор специализированных компонентов, которые улучшают
процесс колоризации изображений, делая его более реалистичным, адаптивным и
управляемым. Каждый компонент решает определенную задачу и может использоваться
как в составе полной системы, так и самостоятельно.

Основные компоненты:
- GuideNet: Советник по цветам на основе семантического анализа изображения
- Discriminator: GAN-дискриминатор с системой наград для реалистичной колоризации
- StyleTransfer: Компонент переноса цветовых стилей между изображениями
- MemoryBank: Самообучающаяся база знаний цветов для различных объектов и сцен
- UncertaintyEstimation: Оценка неопределенности при выборе цветов
- FewShotAdapter: Адаптация между доменами и датасетами с малым количеством примеров

Преимущества для колоризации:
- Более реалистичные и естественные цвета благодаря множеству специализированных модулей
- Адаптивность к различным типам изображений и художественным стилям
- Возможность интерактивного управления и тонкой настройки процесса колоризации
- Самообучение и улучшение результатов с накоплением опыта
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Any

# Импортируем все модули
from .guide_net import GuideNet, create_guide_net
from .discriminator import MotivationalDiscriminator, create_motivational_discriminator
from .style_transfer import StyleTransferModule, create_style_transfer_module
from .memory_bank import MemoryBankModule, create_memory_bank_module
from .uncertainty_estimation import (
    MCDropoutUncertainty, ProbabilisticColorizer, UncertaintyGuidedColorizer,
    create_uncertainty_estimator
)
from .few_shot_adapter import AdaptableColorizer, MetaLearningAdapter, create_few_shot_adapter


class ColorizationModulesManager:
    """
    Менеджер для управления и интеграции всех модулей колоризации.
    
    Обеспечивает единый интерфейс для создания, настройки и взаимодействия
    между различными модулями колоризации.
    
    Args:
        config (Dict): Конфигурация модулей
        colorizer (nn.Module, optional): Базовая модель колоризации
        device (torch.device, optional): Устройство для вычислений
    """
    def __init__(self, config: Dict, colorizer: Optional[nn.Module] = None, device=None):
        self.config = config
        self.colorizer = colorizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Словарь для хранения модулей
        self.modules = {}
        
        # Создаем модули на основе конфигурации
        self._create_modules()
        
    def _create_modules(self):
        """
        Создает модули на основе конфигурации.
        """
        # Проверяем наличие конфигураций для каждого модуля
        module_configs = {
            'guide_net': self.config.get('guide_net', {}),
            'discriminator': self.config.get('discriminator', {}),
            'style_transfer': self.config.get('style_transfer', {}),
            'memory_bank': self.config.get('memory_bank', {}),
            'uncertainty_estimation': self.config.get('uncertainty_estimation', {}),
            'few_shot_adapter': self.config.get('few_shot_adapter', {})
        }
        
        # Проверяем, какие модули нужно включить
        enabled_modules = self.config.get('enabled_modules', [])
        
        # Если список пуст, предполагаем, что все модули включены
        if not enabled_modules:
            enabled_modules = list(module_configs.keys())
        
        # Создаем нужные модули
        for module_name in enabled_modules:
            if module_name not in module_configs:
                print(f"Предупреждение: неизвестный модуль {module_name}")
                continue
                
            # Получаем конфигурацию модуля
            module_config = module_configs[module_name]
            
            # Создаем модуль в зависимости от его типа
            if module_name == 'guide_net' and module_config.get('enabled', True):
                self.modules['guide_net'] = create_guide_net(module_config)
                
            elif module_name == 'discriminator' and module_config.get('enabled', True):
                self.modules['discriminator'] = create_motivational_discriminator(module_config)
                
            elif module_name == 'style_transfer' and module_config.get('enabled', True):
                self.modules['style_transfer'] = create_style_transfer_module(module_config)
                
            elif module_name == 'memory_bank' and module_config.get('enabled', True):
                self.modules['memory_bank'] = create_memory_bank_module(module_config)
                
            elif module_name == 'uncertainty_estimation' and module_config.get('enabled', True) and self.colorizer is not None:
                self.modules['uncertainty_estimation'] = create_uncertainty_estimator(
                    colorizer=self.colorizer,
                    config=module_config
                )
                
            elif module_name == 'few_shot_adapter' and module_config.get('enabled', True) and self.colorizer is not None:
                self.modules['few_shot_adapter'] = create_few_shot_adapter(
                    colorizer=self.colorizer,
                    config=module_config
                )
        
        # Перемещаем модули на нужное устройство
        for name, module in self.modules.items():
            if isinstance(module, nn.Module):
                self.modules[name] = module.to(self.device)
    
    def get_module(self, name: str) -> Optional[nn.Module]:
        """
        Возвращает модуль по имени.
        
        Args:
            name (str): Имя модуля
            
        Returns:
            Optional[nn.Module]: Модуль или None, если модуль не найден
        """
        return self.modules.get(name)
    
    def integrate_with_colorizer(self, colorizer: nn.Module) -> nn.Module:
        """
        Интегрирует модули с колоризатором.
        
        Args:
            colorizer (nn.Module): Базовая модель колоризации
            
        Returns:
            nn.Module: Интегрированная модель колоризации
        """
        # Сохраняем колоризатор
        self.colorizer = colorizer
        
        # Если есть адаптер для few-shot обучения, интегрируем его
        if 'few_shot_adapter' in self.modules:
            colorizer = self.modules['few_shot_adapter']
        
        # Если есть модуль оценки неопределенности, интегрируем его
        if 'uncertainty_estimation' in self.modules:
            if isinstance(self.modules['uncertainty_estimation'], UncertaintyGuidedColorizer):
                # Интегрируем с GuideNet, если он есть
                if 'guide_net' in self.modules:
                    self.modules['uncertainty_estimation'].set_guide_net(self.modules['guide_net'])
                
                # Интегрируем с MemoryBank, если он есть
                if 'memory_bank' in self.modules:
                    self.modules['uncertainty_estimation'].set_memory_bank(self.modules['memory_bank'])
                
                # Устанавливаем как основной колоризатор
                colorizer = self.modules['uncertainty_estimation']
        
        # Возвращаем интегрированную модель
        return colorizer
    
    def apply_style_transfer(self, grayscale_image: torch.Tensor, style_reference: Optional[torch.Tensor] = None,
                          style_name: Optional[str] = None, alpha: float = 1.0) -> Dict:
        """
        Применяет перенос стиля к изображению.
        
        Args:
            grayscale_image (torch.Tensor): Входное ЧБ изображение [B, 1, H, W]
            style_reference (torch.Tensor, optional): Референсное стилевое изображение [B, 3, H, W]
            style_name (str, optional): Имя стиля из словаря
            alpha (float): Интенсивность переноса стиля (0.0 - 1.0)
            
        Returns:
            Dict: Результат переноса стиля
        """
        if 'style_transfer' not in self.modules:
            return {'error': 'Модуль переноса стиля не включен'}
        
        # Перемещаем входные данные на нужное устройство
        grayscale_image = grayscale_image.to(self.device)
        if style_reference is not None:
            style_reference = style_reference.to(self.device)
        
        # Применяем перенос стиля
        return self.modules['style_transfer'].apply_style_transfer(
            grayscale_image, style_reference, style_name, alpha
        )
    
    def get_color_advice(self, grayscale_image: torch.Tensor, reference_image: Optional[torch.Tensor] = None) -> Dict:
        """
        Получает советы по цветам от GuideNet.
        
        Args:
            grayscale_image (torch.Tensor): Входное ЧБ изображение [B, 1, H, W]
            reference_image (torch.Tensor, optional): Референсное изображение [B, 3, H, W]
            
        Returns:
            Dict: Советы по цветам
        """
        if 'guide_net' not in self.modules:
            return {'error': 'Модуль GuideNet не включен'}
        
        # Перемещаем входные данные на нужное устройство
        grayscale_image = grayscale_image.to(self.device)
        if reference_image is not None:
            reference_image = reference_image.to(self.device)
        
        # Получаем советы по цветам
        return self.modules['guide_net'](grayscale_image, reference_image)
    
    def query_memory_bank(self, grayscale_image: torch.Tensor) -> Dict:
        """
        Запрашивает банк памяти для колоризации.
        
        Args:
            grayscale_image (torch.Tensor): Входное ЧБ изображение [B, 1, H, W]
            
        Returns:
            Dict: Результат запроса банка памяти
        """
        if 'memory_bank' not in self.modules:
            return {'error': 'Модуль банка памяти не включен'}
        
        # Перемещаем входные данные на нужное устройство
        grayscale_image = grayscale_image.to(self.device)
        
        # Запрашиваем банк памяти
        return self.modules['memory_bank'](grayscale_image)
    
    def discriminate(self, fake_image: torch.Tensor, real_image: Optional[torch.Tensor] = None,
                   semantics: Optional[torch.Tensor] = None) -> Dict:
        """
        Оценивает реалистичность изображения с помощью дискриминатора.
        
        Args:
            fake_image (torch.Tensor): Сгенерированное изображение [B, C, H, W]
            real_image (torch.Tensor, optional): Реальное изображение [B, C, H, W]
            semantics (torch.Tensor, optional): Семантическая информация [B, S, H, W]
            
        Returns:
            Dict: Результаты дискриминации
        """
        if 'discriminator' not in self.modules:
            return {'error': 'Модуль дискриминатора не включен'}
        
        # Перемещаем входные данные на нужное устройство
        fake_image = fake_image.to(self.device)
        if real_image is not None:
            real_image = real_image.to(self.device)
        if semantics is not None:
            semantics = semantics.to(self.device)
        
        # Оцениваем реалистичность
        return self.modules['discriminator'](fake_image, real_image, semantics)
    
    def adapt_to_domain(self, support_images: torch.Tensor, support_labels: Optional[torch.Tensor] = None,
                       num_steps: int = 100, learning_rate: float = 1e-4) -> Dict:
        """
        Адаптирует модель к новому домену с помощью few-shot обучения.
        
        Args:
            support_images (torch.Tensor): Опорные изображения [N, C, H, W]
            support_labels (torch.Tensor, optional): Метки опорных изображений [N]
            num_steps (int): Количество шагов обучения
            learning_rate (float): Скорость обучения
            
        Returns:
            Dict: Результаты адаптации
        """
        if 'few_shot_adapter' not in self.modules:
            return {'error': 'Модуль few-shot адаптера не включен'}
        
        # Перемещаем входные данные на нужное устройство
        support_images = support_images.to(self.device)
        if support_labels is not None:
            support_labels = support_labels.to(self.device)
        
        # Адаптируем модель
        return self.modules['few_shot_adapter'].few_shot_adapt(
            support_images, support_labels, num_steps, learning_rate
        )
    
    def estimate_uncertainty(self, grayscale_image: torch.Tensor) -> Dict:
        """
        Оценивает неопределенность при колоризации.
        
        Args:
            grayscale_image (torch.Tensor): Входное ЧБ изображение [B, 1, H, W]
            
        Returns:
            Dict: Результаты оценки неопределенности
        """
        if 'uncertainty_estimation' not in self.modules:
            return {'error': 'Модуль оценки неопределенности не включен'}
        
        # Перемещаем входные данные на нужное устройство
        grayscale_image = grayscale_image.to(self.device)
        
        # Оцениваем неопределенность
        return self.modules['uncertainty_estimation'](grayscale_image)
    
    def get_module_stats(self) -> Dict:
        """
        Возвращает статистики всех модулей.
        
        Returns:
            Dict: Статистики модулей
        """
        stats = {}
        
        # Собираем статистики от каждого модуля, если они доступны
        for name, module in self.modules.items():
            if hasattr(module, 'get_stats'):
                stats[name] = module.get_stats()
                
        return stats


# Функция для создания менеджера модулей
def create_modules_manager(config: Dict, colorizer: Optional[nn.Module] = None, device=None) -> ColorizationModulesManager:
    """
    Создает менеджер модулей колоризации.
    
    Args:
        config (Dict): Конфигурация модулей
        colorizer (nn.Module, optional): Базовая модель колоризации
        device (torch.device, optional): Устройство для вычислений
        
    Returns:
        ColorizationModulesManager: Менеджер модулей
    """
    return ColorizationModulesManager(config, colorizer, device)


# Экспортируем все публичные классы и функции
__all__ = [
    # Классы модулей
    'GuideNet', 'MotivationalDiscriminator', 'StyleTransferModule', 
    'MemoryBankModule', 'MCDropoutUncertainty', 'ProbabilisticColorizer',
    'UncertaintyGuidedColorizer', 'AdaptableColorizer', 'MetaLearningAdapter',
    'ColorizationModulesManager',
    
    # Функции создания модулей
    'create_guide_net', 'create_motivational_discriminator',
    'create_style_transfer_module', 'create_memory_bank_module',
    'create_uncertainty_estimator', 'create_few_shot_adapter',
    'create_modules_manager'
]