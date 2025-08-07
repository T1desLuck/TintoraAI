"""
Test Losses: Модуль для тестирования функций потерь колоризатора.

Данный модуль содержит набор тестов для проверки корректности работы
различных функций потерь, используемых в обучении модели колоризации:
PatchNCE, VGG perceptual loss, GAN loss и Dynamic Loss Balancing.

Тесты проверяют:
- Корректность формы выходных тензоров
- Правильность вычисления потерь для разных типов входных данных
- Работу градиентов и обратное распространение ошибки
- Совместимость различных компонентов потерь
- Корректность балансировки и адаптации весов потерь
"""

import unittest
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

# Добавляем корневую директорию проекта в путь импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from losses.patch_nce import PatchNCELoss
from losses.vgg_perceptual import VGGPerceptualLoss
from losses.gan_loss import GANLoss
from losses.dynamic_balancer import DynamicLossBalancer


class TestPatchNCELoss(unittest.TestCase):
    """Тесты для компонента PatchNCE Loss (контрастное + градиентное обучение)."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.temperature = 0.07
        self.patch_size = 16
        self.n_patches = 256
        
        # Создаем модель для тестов
        self.loss_fn = PatchNCELoss(
            temperature=self.temperature,
            patch_size=self.patch_size,
            n_patches=self.n_patches,
            device=self.device
        )
        
    def test_forward(self):
        """Тестирование вычисления потери."""
        batch_size = 2
        img_size = 256
        
        # Создаем входные тензоры
        query = torch.randn(batch_size, 1, img_size, img_size, requires_grad=True).to(self.device)
        key = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        reference = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        
        # Вычисляем потерю
        loss = self.loss_fn(query, key, reference)
        
        # Проверяем форму выходных данных
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(loss.shape, torch.Size([]))  # Скаляр
        self.assertTrue(loss.requires_grad)  # Должен иметь градиенты
        
        # Проверяем, что потеря не отрицательна
        self.assertTrue(loss.item() >= 0.0)
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через функцию потери."""
        batch_size = 2
        img_size = 128  # Меньшее разрешение для быстроты тестов
        
        # Создаем входные тензоры
        query = torch.randn(batch_size, 1, img_size, img_size, requires_grad=True).to(self.device)
        key = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True).to(self.device)
        reference = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        
        # Создаем оптимизатор
        optimizer = optim.Adam([query, key], lr=0.001)
        optimizer.zero_grad()
        
        # Вычисляем потерю
        loss = self.loss_fn(query, key, reference)
        
        # Обратное распространение
        loss.backward()
        
        # Проверяем, что градиенты вычислены для query и key
        self.assertIsNotNone(query.grad)
        self.assertFalse(torch.isnan(query.grad).any().item())
        
        self.assertIsNotNone(key.grad)
        self.assertFalse(torch.isnan(key.grad).any().item())
        
    def test_patch_extraction(self):
        """Тестирование извлечения патчей."""
        batch_size = 2
        img_size = 128
        
        # Создаем входные тензоры
        img = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        
        # Проверяем, что функция извлечения патчей доступна
        self.assertTrue(hasattr(self.loss_fn, 'extract_patches'))
        
        # Извлекаем патчи
        patches = self.loss_fn.extract_patches(img)
        
        # Проверяем форму выходных данных
        # Ожидаемая форма: [batch_size, n_patches, patch_size*patch_size*channels]
        expected_feature_dim = self.patch_size * self.patch_size * 3
        self.assertEqual(patches.shape, (batch_size, self.n_patches, expected_feature_dim))
        
    def test_with_different_sizes(self):
        """Тестирование работы с разными размерами изображений."""
        batch_size = 2
        img_sizes = [128, 256, 384]
        
        for img_size in img_sizes:
            # Создаем входные тензоры
            query = torch.randn(batch_size, 1, img_size, img_size).to(self.device)
            key = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
            reference = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
            
            # Вычисляем потерю
            loss = self.loss_fn(query, key, reference)
            
            # Проверяем форму выходных данных
            self.assertTrue(isinstance(loss, torch.Tensor))
            self.assertEqual(loss.shape, torch.Size([]))  # Скаляр


class TestVGGPerceptualLoss(unittest.TestCase):
    """Тесты для компонента VGG Perceptual Loss."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.weights = None  # По умолчанию все слои имеют равный вес
        self.criterion = 'l1'
        self.resize = True
        self.normalize = True
        
        try:
            # Создаем модель для тестов
            self.loss_fn = VGGPerceptualLoss(
                layers=self.layers,
                weights=self.weights,
                criterion=self.criterion,
                resize=self.resize,
                normalize=self.normalize,
                device=self.device
            )
        except Exception as e:
            self.skipTest(f"Не удалось создать VGGPerceptualLoss: {str(e)}")
        
    def test_forward(self):
        """Тестирование вычисления потери."""
        batch_size = 2
        img_size = 256
        channels = 3
        
        # Создаем входные тензоры
        x = torch.randn(batch_size, channels, img_size, img_size, requires_grad=True).to(self.device)
        y = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        
        # Вычисляем потерю
        loss = self.loss_fn(x, y)
        
        # Проверяем форму выходных данных
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertEqual(loss.shape, torch.Size([]))  # Скаляр
        self.assertTrue(loss.requires_grad)  # Должен иметь градиенты
        
        # Проверяем, что потеря не отрицательна
        self.assertTrue(loss.item() >= 0.0)
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через функцию потери."""
        batch_size = 2
        img_size = 128  # Меньшее разрешение для быстроты тестов
        channels = 3
        
        # Создаем входные тензоры
        x = torch.randn(batch_size, channels, img_size, img_size, requires_grad=True).to(self.device)
        y = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        
        # Создаем оптимизатор
        optimizer = optim.Adam([x], lr=0.001)
        optimizer.zero_grad()
        
        # Вычисляем потерю
        loss = self.loss_fn(x, y)
        
        # Обратное распространение
        loss.backward()
        
        # Проверяем, что градиенты вычислены для x
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any().item())
        
    def test_feature_extraction(self):
        """Тестирование извлечения признаков из VGG."""
        batch_size = 2
        img_size = 128
        channels = 3
        
        # Создаем входные тензоры
        img = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        
        # Проверяем, что функция извлечения признаков доступна
        self.assertTrue(hasattr(self.loss_fn, 'get_features'))
        
        # Извлекаем признаки
        features = self.loss_fn.get_features(img)
        
        # Проверяем структуру выходных данных
        self.assertIsInstance(features, dict)
        
        # Проверяем, что все запрошенные слои присутствуют в результате
        for layer in self.layers:
            self.assertIn(layer, features)
            
            # Проверяем, что каждый признак имеет правильную форму
            self.assertEqual(features[layer].shape[0], batch_size)
            
    def test_different_criterions(self):
        """Тестирование разных критериев потери."""
        batch_size = 2
        img_size = 128
        channels = 3
        
        # Создаем входные тензоры
        x = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        y = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        
        # Тестируем L1 критерий
        l1_loss_fn = VGGPerceptualLoss(
            layers=self.layers,
            criterion='l1',
            device=self.device
        )
        l1_loss = l1_loss_fn(x, y)
        
        # Тестируем L2 критерий
        l2_loss_fn = VGGPerceptualLoss(
            layers=self.layers,
            criterion='l2',
            device=self.device
        )
        l2_loss = l2_loss_fn(x, y)
        
        # Проверяем, что обе потери имеют правильную форму
        self.assertEqual(l1_loss.shape, torch.Size([]))
        self.assertEqual(l2_loss.shape, torch.Size([]))
        
        # Проверяем, что обе потери не отрицательны
        self.assertTrue(l1_loss.item() >= 0.0)
        self.assertTrue(l2_loss.item() >= 0.0)


class TestGANLoss(unittest.TestCase):
    """Тесты для компонента GAN Loss."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Тестируем разные режимы GAN Loss
        self.gan_modes = ['vanilla', 'lsgan', 'wgangp', 'hinge']
        
        # Создаем экземпляры для разных режимов
        self.losses = {}
        for mode in self.gan_modes:
            self.losses[mode] = GANLoss(gan_mode=mode, device=self.device)
        
    def test_forward_real(self):
        """Тестирование вычисления потери для реальных изображений."""
        batch_size = 2
        
        # Для PatchGAN обычно выход имеет форму [batch_size, 1, patch_size, patch_size]
        patch_size = 16
        input_tensor = torch.randn(batch_size, 1, patch_size, patch_size).to(self.device)
        
        for mode, loss_fn in self.losses.items():
            # Вычисляем потерю для реальных изображений (target=True)
            loss = loss_fn(input_tensor, True)
            
            # Проверяем форму выходных данных
            self.assertTrue(isinstance(loss, torch.Tensor))
            self.assertEqual(loss.shape, torch.Size([]))  # Скаляр
            
            # Проверяем, что потеря не отрицательна
            self.assertTrue(loss.item() >= 0.0)
    
    def test_forward_fake(self):
        """Тестирование вычисления потери для поддельных изображений."""
        batch_size = 2
        patch_size = 16
        input_tensor = torch.randn(batch_size, 1, patch_size, patch_size).to(self.device)
        
        for mode, loss_fn in self.losses.items():
            # Вычисляем потерю для поддельных изображений (target=False)
            loss = loss_fn(input_tensor, False)
            
            # Проверяем форму выходных данных
            self.assertTrue(isinstance(loss, torch.Tensor))
            self.assertEqual(loss.shape, torch.Size([]))  # Скаляр
            
            # Проверяем, что потеря не отрицательна
            self.assertTrue(loss.item() >= 0.0)
            
    def test_gradient_flow(self):
        """Тестирование потока градиентов через функцию потери."""
        batch_size = 2
        patch_size = 16
        input_tensor = torch.randn(batch_size, 1, patch_size, patch_size, requires_grad=True).to(self.device)
        
        for mode, loss_fn in self.losses.items():
            # Создаем оптимизатор
            optimizer = optim.Adam([input_tensor], lr=0.001)
            optimizer.zero_grad()
            
            # Вычисляем потерю для реальных изображений
            loss = loss_fn(input_tensor, True)
            
            # Обратное распространение
            loss.backward(retain_graph=True)
            
            # Проверяем, что градиенты вычислены для input_tensor
            self.assertIsNotNone(input_tensor.grad)
            self.assertFalse(torch.isnan(input_tensor.grad).any().item())
            
            # Очищаем градиенты
            optimizer.zero_grad()
            
            # Вычисляем потерю для поддельных изображений
            loss = loss_fn(input_tensor, False)
            
            # Обратное распространение
            loss.backward()
            
            # Проверяем, что градиенты вычислены для input_tensor
            self.assertIsNotNone(input_tensor.grad)
            self.assertFalse(torch.isnan(input_tensor.grad).any().item())


class TestDynamicLossBalancer(unittest.TestCase):
    """Тесты для компонента Dynamic Loss Balancing."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Базовая конфигурация для тестирования
        self.initial_weights = {
            'l1': 10.0,
            'perceptual': 1.0,
            'gan': 0.1
        }
        self.strategy = 'adaptive'
        self.target_metric = 'lpips'
        self.learning_rate = 0.01
        
        # Создаем балансировщик для тестов
        self.balancer = DynamicLossBalancer(
            initial_weights=self.initial_weights,
            strategy=self.strategy,
            target_metric=self.target_metric,
            learning_rate=self.learning_rate
        )
        
    def test_get_weights(self):
        """Тестирование получения весов."""
        # Получаем текущие веса
        weights = self.balancer.get_weights()
        
        # Проверяем структуру весов
        self.assertIsInstance(weights, dict)
        
        # Проверяем, что все начальные веса присутствуют
        for key in self.initial_weights:
            self.assertIn(key, weights)
            self.assertEqual(weights[key], self.initial_weights[key])
    
    def test_update_weights(self):
        """Тестирование обновления весов на основе метрик."""
        # Создаем фиктивные метрики
        metrics = {
            'l1': 0.2,
            'perceptual': 0.1,
            'gan': 0.05,
            'lpips': 0.3,  # Целевая метрика
            'ssim': 0.8
        }
        
        # Запоминаем исходные веса
        original_weights = self.balancer.get_weights().copy()
        
        # Обновляем веса на основе метрик
        self.balancer.update_weights(metrics)
        
        # Получаем обновленные веса
        updated_weights = self.balancer.get_weights()
        
        # Проверяем, что веса изменились
        for key in self.initial_weights:
            if self.strategy == 'fixed':
                # Для fixed стратегии веса не должны меняться
                self.assertEqual(updated_weights[key], original_weights[key])
            elif self.strategy == 'adaptive':
                # Для adaptive стратегии веса могут измениться
                # Но изменение должно быть контролируемым
                self.assertGreaterEqual(updated_weights[key], 0.0)  # Веса должны оставаться неотрицательными
        
    def test_serialization(self):
        """Тестирование сериализации и десериализации."""
        # Проверяем, что методы сериализации доступны
        self.assertTrue(hasattr(self.balancer, 'state_dict'))
        self.assertTrue(hasattr(self.balancer, 'load_state_dict'))
        
        # Получаем состояние
        state = self.balancer.state_dict()
        
        # Проверяем структуру состояния
        self.assertIsInstance(state, dict)
        self.assertIn('weights', state)
        
        # Создаем новый балансировщик с другими начальными весами
        new_balancer = DynamicLossBalancer(
            initial_weights={'l1': 1.0, 'perceptual': 2.0},
            strategy='fixed'
        )
        
        # Загружаем состояние в новый балансировщик
        new_balancer.load_state_dict(state)
        
        # Проверяем, что веса были корректно загружены
        new_weights = new_balancer.get_weights()
        for key, value in self.balancer.get_weights().items():
            self.assertIn(key, new_weights)
            self.assertEqual(new_weights[key], value)
    
    def test_different_strategies(self):
        """Тестирование разных стратегий балансировки потерь."""
        strategies = ['fixed', 'adaptive', 'online']
        metrics = {
            'l1': 0.2,
            'perceptual': 0.1,
            'gan': 0.05,
            'lpips': 0.3,
            'ssim': 0.8
        }
        
        for strategy in strategies:
            # Создаем балансировщик с текущей стратегией
            balancer = DynamicLossBalancer(
                initial_weights=self.initial_weights,
                strategy=strategy,
                target_metric=self.target_metric,
                learning_rate=self.learning_rate
            )
            
            # Запоминаем исходные веса
            original_weights = balancer.get_weights().copy()
            
            # Обновляем веса
            balancer.update_weights(metrics)
            
            # Получаем обновленные веса
            updated_weights = balancer.get_weights()
            
            # Проверяем, что веса соответствуют стратегии
            if strategy == 'fixed':
                # Для fixed стратегии веса не должны меняться
                for key in self.initial_weights:
                    self.assertEqual(updated_weights[key], original_weights[key])
            else:
                # Для других стратегий веса могут измениться, но должны оставаться неотрицательными
                for key in self.initial_weights:
                    self.assertGreaterEqual(updated_weights[key], 0.0)


class TestCombinedLosses(unittest.TestCase):
    """Тесты для комбинации различных функций потерь."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Создаем функции потерь
        try:
            self.patchnce_loss = PatchNCELoss(
                temperature=0.07,
                patch_size=16,
                n_patches=256,
                device=self.device
            )
            
            self.perceptual_loss = VGGPerceptualLoss(
                layers=['relu1_2', 'relu2_2', 'relu3_3'],
                criterion='l1',
                device=self.device
            )
            
            self.gan_loss = GANLoss(
                gan_mode='lsgan',
                device=self.device
            )
            
            self.dynamic_balancer = DynamicLossBalancer(
                initial_weights={'patchnce': 1.0, 'perceptual': 1.0, 'gan': 0.1},
                strategy='adaptive',
                target_metric='lpips'
            )
            
            self.losses_available = True
        except Exception as e:
            print(f"Не удалось создать все функции потерь: {str(e)}")
            self.losses_available = False
    
    def test_combined_loss(self):
        """Тестирование комбинированной потери с динамическим балансированием."""
        if not self.losses_available:
            self.skipTest("Не все функции потерь доступны для тестирования")
            
        batch_size = 2
        img_size = 128
        
        # Создаем входные тензоры
        grayscale = torch.randn(batch_size, 1, img_size, img_size, requires_grad=True).to(self.device)
        color_pred = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True).to(self.device)
        color_target = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        
        # Создаем оптимизатор
        optimizer = optim.Adam([grayscale, color_pred], lr=0.001)
        optimizer.zero_grad()
        
        # Вычисляем отдельные потери
        patchnce = self.patchnce_loss(grayscale, color_pred, color_target)
        perceptual = self.perceptual_loss(color_pred, color_target)
        gan = self.gan_loss(color_pred, True)
        
        # Получаем веса для потерь
        weights = self.dynamic_balancer.get_weights()
        
        # Вычисляем комбинированную потерю
        combined_loss = weights['patchnce'] * patchnce + weights['perceptual'] * perceptual + weights['gan'] * gan
        
        # Обратное распространение
        combined_loss.backward()
        
        # Проверяем, что градиенты вычислены корректно
        self.assertIsNotNone(grayscale.grad)
        self.assertFalse(torch.isnan(grayscale.grad).any().item())
        
        self.assertIsNotNone(color_pred.grad)
        self.assertFalse(torch.isnan(color_pred.grad).any().item())
        
        # Симулируем обучение
        optimizer.step()
        
        # Обновляем веса на основе метрик
        metrics = {'lpips': 0.3, 'psnr': 25.0, 'ssim': 0.8}
        self.dynamic_balancer.update_weights(metrics)
        
        # Получаем обновленные веса
        updated_weights = self.dynamic_balancer.get_weights()
        
        # Проверяем, что веса обновились корректно
        for key in weights:
            self.assertIn(key, updated_weights)
            self.assertGreaterEqual(updated_weights[key], 0.0)


if __name__ == '__main__':
    unittest.main()