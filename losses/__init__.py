"""
TintoraAI Losses: Набор специализированных функций потерь для модели колоризации.

Данный модуль объединяет все функции потерь, используемые в проекте TintoraAI,
и предоставляет удобные интерфейсы для их создания и использования. Функции потерь 
специально разработаны для задачи колоризации изображений и включают компоненты для
обеспечения реалистичности цветов, сохранения текстур и деталей, и поддержания 
цветовой согласованности.

Основные компоненты:
- PatchNCE Loss: Контрастное и градиентное обучение для улучшения локальных цветовых паттернов
- VGG Perceptual Loss: Расширенная перцептивная функция потерь для сохранения семантики цветов
- GAN Loss: Специализированная GAN функция потерь с системой "наград и наказаний"
- Dynamic Loss Balancer: Адаптивное взвешивание различных функций потерь в процессе обучения

Все функции потерь могут быть легко настроены через конфигурационные файлы,
что позволяет адаптировать систему для различных задач колоризации без изменения кода.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Tuple, Any

# Импортируем все функции потерь
from .patch_nce import MultiScalePatchNCELoss, MultiScalePatchwiseLoss, create_patch_nce_loss
from .vgg_perceptual import VGGEncoder, StyleLoss, ContentLoss, VGGPerceptualLoss, EnhancedVGGPerceptual, create_vgg_perceptual_loss
from .gan_loss import GANLoss, AdversarialColorLoss, MotivationalColorLoss, RewardPenaltyManager, create_gan_loss
from .dynamic_balancer import LossHistoryTracker, DynamicWeightBalancer, AdaptiveLossBalancer, create_dynamic_loss_balancer


class CompositeLoss(nn.Module):
    """
    Композитная функция потерь, объединяющая все компоненты с адаптивным взвешиванием.
    
    Args:
        loss_config (Dict): Конфигурация функций потерь
        device (torch.device): Устройство для вычислений
    """
    def __init__(self, loss_config: Dict, device: torch.device = None):
        super(CompositeLoss, self).__init__()
        
        self.loss_config = loss_config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Флаги для включения/выключения различных компонентов
        self.use_patch_nce = loss_config.get('use_patch_nce', True)
        self.use_vgg_perceptual = loss_config.get('use_vgg_perceptual', True)
        self.use_gan = loss_config.get('use_gan', True)
        self.use_dynamic_balancer = loss_config.get('use_dynamic_balancer', True)
        
        # Инициализация компонентов функций потерь
        if self.use_patch_nce:
            self.patch_nce_loss = create_patch_nce_loss(loss_config.get('patch_nce', {}))
            
        if self.use_vgg_perceptual:
            self.vgg_loss = create_vgg_perceptual_loss(loss_config.get('vgg_perceptual', {}))
            
        if self.use_gan:
            self.gan_loss = create_gan_loss(loss_config.get('gan', {}))
            
        # Список имен всех используемых функций потерь
        loss_names = []
        
        if self.use_patch_nce:
            loss_names.append('patch_nce')
            
        if self.use_vgg_perceptual:
            loss_names.append('vgg_perceptual')
            
        if self.use_gan:
            loss_names.append('gan')
            
        # Инициализируем Dynamic Loss Balancer
        if self.use_dynamic_balancer and len(loss_names) > 1:
            self.dynamic_balancer = create_dynamic_loss_balancer(
                loss_names, 
                loss_config.get('dynamic_balancer', {})
            )
        else:
            self.use_dynamic_balancer = False
            
        # Регистрируем счетчики для статистики
        self.register_buffer('iteration_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('epoch_count', torch.zeros(1, dtype=torch.long))
        
    def set_epoch(self, epoch: int):
        """
        Устанавливает текущую эпоху для планирования весов.
        
        Args:
            epoch (int): Номер текущей эпохи
        """
        self.epoch_count[0] = epoch
        
        if self.use_dynamic_balancer:
            self.dynamic_balancer.set_epoch(epoch)
            
    def forward(self, 
                model_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                discriminator: Optional[nn.Module] = None,
                features: Optional[Dict[str, List[torch.Tensor]]] = None,
                is_generator_step: bool = True) -> Dict[str, torch.Tensor]:
        """
        Вычисляет композитную функцию потерь.
        
        Args:
            model_outputs (Dict[str, torch.Tensor]): Выходы модели колоризации
            targets (Dict[str, torch.Tensor]): Целевые значения
            discriminator (nn.Module, optional): Дискриминатор для GAN потерь
            features (Dict[str, List[torch.Tensor]], optional): Промежуточные признаки модели
            is_generator_step (bool): Является ли текущий шаг шагом генератора или дискриминатора
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с результатами вычисления потерь
        """
        # Увеличиваем счетчик итераций
        self.iteration_count += 1
        
        # Извлекаем необходимые данные из входов
        predicted_color = model_outputs['colorized']
        target_color = targets['color']
        grayscale = targets['grayscale']
        
        # Словарь для хранения результатов потерь
        losses = {}
        
        # Вычисляем PatchNCE потерю, если используется
        if self.use_patch_nce and is_generator_step:
            # Получаем промежуточные признаки, если они предоставлены
            encoder_features = features.get('encoder', []) if features else []
            decoder_features = features.get('decoder', []) if features else []
            
            if encoder_features and decoder_features:
                # Вычисляем потерю с промежуточными признаками
                patch_features = {
                    'encoder': encoder_features,
                    'decoder': decoder_features
                }
                
                patch_nce_result = self.patch_nce_loss(patch_features, predicted_color, target_color)
                losses['patch_nce'] = patch_nce_result['total_loss']
                
                # Добавляем компоненты потери для подробной статистики
                losses['patch_nce_components'] = {
                    'nce': patch_nce_result['nce_loss'],
                    'gradient': patch_nce_result.get('gradient_loss', torch.tensor(0.0, device=self.device)),
                    'consistency': patch_nce_result.get('consistency_loss', torch.tensor(0.0, device=self.device))
                }
                
        # Вычисляем VGG Perceptual потерю, если используется
        if self.use_vgg_perceptual and is_generator_step:
            vgg_result = self.vgg_loss(predicted_color, target_color, grayscale)
            losses['vgg_perceptual'] = vgg_result['total_loss']
            
            # Добавляем компоненты потери для подробной статистики
            losses['vgg_perceptual_components'] = {
                'perceptual': vgg_result['perceptual_loss'],
                'pixel': vgg_result.get('pixel_loss', torch.tensor(0.0, device=self.device)),
                'gradient': vgg_result.get('gradient_loss', torch.tensor(0.0, device=self.device)),
                'frequency': vgg_result.get('frequency_loss', torch.tensor(0.0, device=self.device))
            }
            
        # Вычисляем GAN потерю, если используется
        if self.use_gan and discriminator is not None:
            # Получаем промежуточные признаки для GAN, если они предоставлены
            fake_features = features.get('generator', []) if features else None
            real_features = features.get('target', []) if features else None
            
            # Вычисляем GAN потерю
            gan_result = self.gan_loss(
                discriminator, 
                predicted_color, 
                target_color, 
                grayscale,
                fake_features, 
                real_features, 
                is_generator_step
            )
            
            losses['gan'] = gan_result['loss']
            
            # Добавляем компоненты потери для подробной статистики
            losses['gan_components'] = {
                'gan_loss': gan_result['gan_loss'],
                'quality_score': gan_result.get('quality_score', torch.tensor(0.0, device=self.device))
            }
            
            if 'color_loss' in gan_result:
                losses['gan_components']['color_loss'] = gan_result['color_loss']
                
        # Если включен динамический балансировщик, применяем его
        if self.use_dynamic_balancer and is_generator_step:
            # Словарь для основных функций потерь
            main_losses = {name: loss for name, loss in losses.items() 
                          if name in ['patch_nce', 'vgg_perceptual', 'gan']}
            
            # Градиенты для динамического балансирования (если доступны)
            gradients = None  # В реальной реализации можно добавить отслеживание градиентов
            
            # Применяем балансировщик
            balancer_result = self.dynamic_balancer(main_losses, update_stats=True, gradients=gradients)
            
            # Общая потеря
            total_loss = balancer_result['total_loss']
            
            # Веса для отчетности
            losses['weights'] = balancer_result['weights']
        else:
            # Если не используем балансировщик, просто суммируем основные потери
            total_loss = sum(losses.get(name, 0) for name in ['patch_nce', 'vgg_perceptual', 'gan'] 
                           if name in losses)
        
        # Добавляем общую потерю к результатам
        losses['total'] = total_loss
        
        return losses


def create_loss_function(config: Dict, device: torch.device = None) -> nn.Module:
    """
    Создает функцию потерь на основе конфигурации.
    
    Args:
        config (Dict): Конфигурация функции потерь
        device (torch.device): Устройство для вычислений
        
    Returns:
        nn.Module: Функция потерь
    """
    return CompositeLoss(config, device)


# Экспортируем все публичные классы и функции
__all__ = [
    # Основные классы потерь
    'MultiScalePatchNCELoss', 'VGGPerceptualLoss', 'EnhancedVGGPerceptual',
    'GANLoss', 'AdversarialColorLoss', 'MotivationalColorLoss',
    'DynamicWeightBalancer', 'AdaptiveLossBalancer', 'CompositeLoss',
    
    # Вспомогательные классы
    'VGGEncoder', 'StyleLoss', 'ContentLoss', 'LossHistoryTracker', 'RewardPenaltyManager',
    
    # Функции для создания потерь
    'create_patch_nce_loss', 'create_vgg_perceptual_loss', 'create_gan_loss',
    'create_dynamic_loss_balancer', 'create_loss_function'
]