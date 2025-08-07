"""
Test Core: Модуль для тестирования core-компонентов архитектуры колоризатора.

Данный модуль содержит набор тестов для проверки корректности работы
основных компонентов архитектуры колоризатора, таких как Swin-UNet,
ViT, FPN, Cross-Attention Bridge и Feature Fusion Module.

Тесты проверяют:
- Корректность формы выходных тензоров
- Работоспособность на различных размерах входных данных
- Совместимость компонентов при их соединении
- Эффективность работы внимания и других ключевых механизмов
- Корректность распространения градиентов для обучения
"""

import unittest
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

# Добавляем корневую директорию проекта в путь импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.swin_unet import SwinUNet
from core.vit_semantic import ViTSemantic
from core.fpn_pyramid import FPNPyramid
from core.cross_attention_bridge import CrossAttentionBridge
from core.feature_fusion import MultiHeadFeatureFusion


class TestSwinUNet(unittest.TestCase):
    """Тесты для компонента Swin-UNet."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.img_size = 256
        self.in_channels = 1  # Входной L-канал для колоризации
        self.out_channels = 2  # Выходные a и b каналы для Lab цветового пространства
        self.patch_size = 4
        self.embed_dim = 96
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
        self.window_size = 8
        
        # Создаем модель для тестов
        self.model = SwinUNet(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            return_intermediate=True  # Нужно для FPN
        ).to(self.device)
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.in_channels, self.img_size, self.img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Проверяем форму выходных данных
        if isinstance(output, tuple) or isinstance(output, list):
            # Проверка промежуточных выходов
            self.assertEqual(len(output), 4)  # Должно быть 4 промежуточных выхода
            final_output = output[-1]
            self.assertEqual(final_output.shape, (batch_size, self.out_channels, self.img_size, self.img_size))
            
            # Проверяем форму промежуточных выходов
            expected_shapes = [
                (batch_size, self.embed_dim, self.img_size // 4, self.img_size // 4),
                (batch_size, self.embed_dim * 2, self.img_size // 8, self.img_size // 8),
                (batch_size, self.embed_dim * 4, self.img_size // 16, self.img_size // 16),
                (batch_size, self.embed_dim * 8, self.img_size // 32, self.img_size // 32)
            ]
            
            for i, feature_map in enumerate(output[:-1]):
                self.assertEqual(feature_map.shape, expected_shapes[i])
        else:
            # Если выход - один тензор
            self.assertEqual(output.shape, (batch_size, self.out_channels, self.img_size, self.img_size))
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через модель."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.in_channels, self.img_size, self.img_size, requires_grad=True).to(self.device)
        target = torch.randn(batch_size, self.out_channels, self.img_size, self.img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Берем последний выход, если модель возвращает промежуточные выходы
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[-1]
            
        # Вычисляем потерю и градиенты
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Проверяем, что градиенты не None и не содержат NaN
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any().item())
    
    def test_varying_input_sizes(self):
        """Тестирование работы с различными размерами входных данных."""
        sizes = [224, 256, 384, 512]
        
        for size in sizes:
            model = SwinUNet(
                img_size=size,
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                embed_dim=self.embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                window_size=self.window_size
            ).to(self.device)
            
            batch_size = 1
            input_tensor = torch.randn(batch_size, self.in_channels, size, size).to(self.device)
            
            # Выполняем прямой проход
            output = model(input_tensor)
            
            # Проверяем форму выходных данных
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[-1]
                
            self.assertEqual(output.shape, (batch_size, self.out_channels, size, size))


class TestViTSemantic(unittest.TestCase):
    """Тесты для компонента ViT Semantic."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.img_size = 256
        self.in_channels = 1
        self.patch_size = 16
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        
        # Создаем модель для тестов
        self.model = ViTSemantic(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads
        ).to(self.device)
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.in_channels, self.img_size, self.img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Ожидаемый размер выходного тензора
        expected_size = (batch_size, self.embed_dim, self.img_size // self.patch_size, self.img_size // self.patch_size)
        
        # Проверяем форму выходных данных
        self.assertEqual(output.shape, expected_size)
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через модель."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.in_channels, self.img_size, self.img_size, requires_grad=True).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Генерируем фиктивный градиент для обратного распространения
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        
        # Проверяем, что градиенты не None и не содержат NaN
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any().item())


class TestFPNPyramid(unittest.TestCase):
    """Тесты для компонента FPN Pyramid."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.in_channels_list = [96, 192, 384, 768]
        self.out_channels = 256
        self.use_pyramid_pooling = True
        
        # Создаем модель для тестов
        self.model = FPNPyramid(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            use_pyramid_pooling=self.use_pyramid_pooling
        ).to(self.device)
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        
        # Создаем входные фичи разных размеров, имитирующие выходы из Swin-UNet
        c1 = torch.randn(batch_size, self.in_channels_list[0], 64, 64).to(self.device)
        c2 = torch.randn(batch_size, self.in_channels_list[1], 32, 32).to(self.device)
        c3 = torch.randn(batch_size, self.in_channels_list[2], 16, 16).to(self.device)
        c4 = torch.randn(batch_size, self.in_channels_list[3], 8, 8).to(self.device)
        
        features = [c1, c2, c3, c4]
        
        # Выполняем прямой проход
        output = self.model(features)
        
        # Проверяем, что выходные данные имеют правильную форму
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], self.out_channels)
        
        # Проверяем размер выходного тензора (должен быть как у первого входа)
        self.assertEqual(output.shape[2], c1.shape[2])
        self.assertEqual(output.shape[3], c1.shape[3])
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через модель."""
        batch_size = 2
        
        # Создаем входные фичи разных размеров
        c1 = torch.randn(batch_size, self.in_channels_list[0], 64, 64, requires_grad=True).to(self.device)
        c2 = torch.randn(batch_size, self.in_channels_list[1], 32, 32, requires_grad=True).to(self.device)
        c3 = torch.randn(batch_size, self.in_channels_list[2], 16, 16, requires_grad=True).to(self.device)
        c4 = torch.randn(batch_size, self.in_channels_list[3], 8, 8, requires_grad=True).to(self.device)
        
        features = [c1, c2, c3, c4]
        
        # Выполняем прямой проход
        output = self.model(features)
        
        # Генерируем фиктивный градиент для обратного распространения
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        
        # Проверяем, что градиенты не None и не содержат NaN
        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertFalse(torch.isnan(feature.grad).any().item())


class TestCrossAttentionBridge(unittest.TestCase):
    """Тесты для компонента Cross Attention Bridge."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.swin_dim = 256
        self.vit_dim = 768
        self.num_heads = 8
        
        # Создаем модель для тестов
        self.model = CrossAttentionBridge(
            swin_dim=self.swin_dim,
            vit_dim=self.vit_dim,
            num_heads=self.num_heads
        ).to(self.device)
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        h, w = 64, 64
        
        # Создаем входные тензоры, имитирующие выходы из FPN и ViT
        fpn_features = torch.randn(batch_size, self.swin_dim, h, w).to(self.device)
        vit_features = torch.randn(batch_size, self.vit_dim, h // 4, w // 4).to(self.device)  # ViT обычно имеет меньшее разрешение
        
        # Выполняем прямой проход
        output = self.model(fpn_features, vit_features)
        
        # Проверяем, что выходные данные имеют правильную форму
        self.assertEqual(output.shape, (batch_size, self.swin_dim, h, w))
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через модель."""
        batch_size = 2
        h, w = 64, 64
        
        # Создаем входные тензоры с требованием градиента
        fpn_features = torch.randn(batch_size, self.swin_dim, h, w, requires_grad=True).to(self.device)
        vit_features = torch.randn(batch_size, self.vit_dim, h // 4, w // 4, requires_grad=True).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(fpn_features, vit_features)
        
        # Генерируем фиктивный градиент для обратного распространения
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        
        # Проверяем, что градиенты не None и не содержат NaN
        self.assertIsNotNone(fpn_features.grad)
        self.assertFalse(torch.isnan(fpn_features.grad).any().item())
        
        self.assertIsNotNone(vit_features.grad)
        self.assertFalse(torch.isnan(vit_features.grad).any().item())


class TestMultiHeadFeatureFusion(unittest.TestCase):
    """Тесты для компонента Multi-Head Feature Fusion."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.in_channels_list = [256, 768]
        self.out_channels = 512
        self.num_heads = 8
        
        # Создаем модель для тестов
        self.model = MultiHeadFeatureFusion(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            num_heads=self.num_heads
        ).to(self.device)
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        h, w = 64, 64
        
        # Создаем входные тензоры разных размерностей
        features = [
            torch.randn(batch_size, self.in_channels_list[0], h, w).to(self.device),
            torch.randn(batch_size, self.in_channels_list[1], h, w).to(self.device)
        ]
        
        # Выполняем прямой проход
        output = self.model(features)
        
        # Проверяем, что выходные данные имеют правильную форму
        self.assertEqual(output.shape, (batch_size, self.out_channels, h, w))
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через модель."""
        batch_size = 2
        h, w = 64, 64
        
        # Создаем входные тензоры с требованием градиента
        features = [
            torch.randn(batch_size, self.in_channels_list[0], h, w, requires_grad=True).to(self.device),
            torch.randn(batch_size, self.in_channels_list[1], h, w, requires_grad=True).to(self.device)
        ]
        
        # Выполняем прямой проход
        output = self.model(features)
        
        # Генерируем фиктивный градиент для обратного распространения
        grad_output = torch.ones_like(output)
        output.backward(grad_output)
        
        # Проверяем, что градиенты не None и не содержат NaN
        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertFalse(torch.isnan(feature.grad).any().item())


class TestFullArchitecture(unittest.TestCase):
    """Тесты для полной архитектуры колоризации, объединяющей все компоненты."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA и выделенную память
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Если тестирование на GPU, убедимся, что есть достаточно памяти
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Базовая конфигурация для тестирования
        self.img_size = 256
        self.batch_size = 2
        
        # Параметры Swin-UNet
        self.in_channels = 1
        self.out_channels = 2
        self.patch_size = 4
        self.swin_embed_dim = 96
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
        self.window_size = 8
        
        # Параметры ViT
        self.vit_patch_size = 16
        self.vit_embed_dim = 768
        self.vit_depth = 12
        self.vit_num_heads = 12
        
        # Параметры FPN
        self.fpn_in_channels = [96, 192, 384, 768]
        self.fpn_out_channels = 256
        
        # Параметры Cross-Attention
        self.ca_num_heads = 8
        
        # Параметры Feature Fusion
        self.ff_in_channels = [256, 768]
        self.ff_out_channels = 512
        self.ff_num_heads = 8
        
    def test_full_architecture_forward(self):
        """Тестирование прямого прохода через полную архитектуру."""
        try:
            # Создаем компоненты архитектуры
            swin_unet = SwinUNet(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                embed_dim=self.swin_embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                window_size=self.window_size,
                return_intermediate=True
            ).to(self.device)
            
            vit_semantic = ViTSemantic(
                img_size=self.img_size,
                patch_size=self.vit_patch_size,
                in_channels=self.in_channels,
                embed_dim=self.vit_embed_dim,
                depth=self.vit_depth,
                num_heads=self.vit_num_heads
            ).to(self.device)
            
            fpn = FPNPyramid(
                in_channels_list=self.fpn_in_channels,
                out_channels=self.fpn_out_channels,
                use_pyramid_pooling=True
            ).to(self.device)
            
            cross_attention = CrossAttentionBridge(
                swin_dim=self.fpn_out_channels,
                vit_dim=self.vit_embed_dim,
                num_heads=self.ca_num_heads
            ).to(self.device)
            
            feature_fusion = MultiHeadFeatureFusion(
                in_channels_list=self.ff_in_channels,
                out_channels=self.ff_out_channels,
                num_heads=self.ff_num_heads
            ).to(self.device)
            
            # Финальный слой для получения выходных каналов
            final_layer = nn.Conv2d(
                self.ff_out_channels,
                self.out_channels,
                kernel_size=1
            ).to(self.device)
            
            # Создаем входной тензор
            input_tensor = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size).to(self.device)
            
            # Выполняем прямой проход через всю архитектуру
            # 1. Swin-UNet обработка
            swin_features = swin_unet(input_tensor)
            
            # 2. ViT обработка
            vit_features = vit_semantic(input_tensor)
            
            # 3. FPN обработка (используем промежуточные выходы Swin-UNet)
            fpn_features = fpn(swin_features[:-1])  # Исключаем последний выход, который является финальным
            
            # 4. Cross-Attention между FPN и ViT
            attended_features = cross_attention(fpn_features, vit_features)
            
            # 5. Слияние признаков
            fused_features = feature_fusion([attended_features, vit_features])
            
            # 6. Финальное предсказание
            output = final_layer(fused_features)
            
            # Проверяем, что выходные данные имеют правильную форму
            self.assertEqual(output.shape, (self.batch_size, self.out_channels, self.img_size, self.img_size))
            
        except Exception as e:
            self.fail(f"Тест полной архитектуры не прошел с ошибкой: {e}")
    
    def test_full_architecture_gradient_flow(self):
        """Тестирование потока градиентов через полную архитектуру."""
        try:
            # Создаем полную модель, которая объединяет все компоненты
            class FullModel(nn.Module):
                def __init__(self, img_size, in_channels, out_channels):
                    super().__init__()
                    
                    # Параметры Swin-UNet
                    self.swin_unet = SwinUNet(
                        img_size=img_size,
                        patch_size=4,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=8,
                        return_intermediate=True
                    )
                    
                    # ViT для семантического понимания
                    self.vit_semantic = ViTSemantic(
                        img_size=img_size,
                        patch_size=16,
                        in_channels=in_channels,
                        embed_dim=768,
                        depth=12,
                        num_heads=12
                    )
                    
                    # FPN с пирамидальным пулингом
                    self.fpn = FPNPyramid(
                        in_channels_list=[96, 192, 384, 768],
                        out_channels=256,
                        use_pyramid_pooling=True
                    )
                    
                    # Мост Cross-Attention
                    self.cross_attention = CrossAttentionBridge(
                        swin_dim=256,
                        vit_dim=768,
                        num_heads=8
                    )
                    
                    # Модуль слияния признаков
                    self.feature_fusion = MultiHeadFeatureFusion(
                        in_channels_list=[256, 768],
                        out_channels=512,
                        num_heads=8
                    )
                    
                    # Финальный слой
                    self.final = nn.Conv2d(512, out_channels, kernel_size=1)
                    
                def forward(self, x):
                    # Swin-UNet обработка
                    swin_features = self.swin_unet(x)
                    
                    # ViT обработка
                    vit_features = self.vit_semantic(x)
                    
                    # FPN обработка
                    fpn_features = self.fpn(swin_features[:-1])
                    
                    # Cross-Attention между FPN и ViT
                    attended_features = self.cross_attention(fpn_features, vit_features)
                    
                    # Слияние признаков
                    fused_features = self.feature_fusion([attended_features, vit_features])
                    
                    # Финальное предсказание
                    output = self.final(fused_features)
                    
                    return {'a': output[:, 0:1], 'b': output[:, 1:2]}
            
            # Создаем модель и переносим на устройство
            model = FullModel(
                img_size=self.img_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels
            ).to(self.device)
            
            # Создаем входной тензор с требованием градиента
            input_tensor = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size, requires_grad=True).to(self.device)
            
            # Создаем целевой тензор
            target_a = torch.randn(self.batch_size, 1, self.img_size, self.img_size).to(self.device)
            target_b = torch.randn(self.batch_size, 1, self.img_size, self.img_size).to(self.device)
            target = {'a': target_a, 'b': target_b}
            
            # Выполняем прямой проход
            output = model(input_tensor)
            
            # Вычисляем потерю
            loss = nn.MSELoss()(output['a'], target['a']) + nn.MSELoss()(output['b'], target['b'])
            
            # Выполняем обратное распространение
            loss.backward()
            
            # Проверяем, что градиенты не None и не содержат NaN для всех параметров модели
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Градиент для {name} равен None")
                    self.assertFalse(torch.isnan(param.grad).any().item(), f"Градиент для {name} содержит NaN")
                    
            # Проверяем, что входной тензор имеет градиенты
            self.assertIsNotNone(input_tensor.grad)
            self.assertFalse(torch.isnan(input_tensor.grad).any().item())
            
        except Exception as e:
            self.fail(f"Тест потока градиентов не прошел с ошибкой: {e}")


if __name__ == '__main__':
    unittest.main()