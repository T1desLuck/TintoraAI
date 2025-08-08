"""
Test Modules: Модуль для тестирования интеллектуальных компонентов колоризатора.

Данный модуль содержит набор тестов для проверки корректности работы
интеллектуальных модулей колоризатора, таких как GuideNet, Discriminator,
Style Transfer, Memory Bank, Uncertainty Estimation и Few-Shot Adapter.

Тесты проверяют:
- Корректность формы выходных тензоров
- Работоспособность системы наград и наказаний
- Правильность процессов обучения и вывода
- Интеграцию между различными модулями
- Эффективность и стабильность специализированных механизмов
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

from modules.guide_net import GuideNet
from modules.discriminator import MotivationalDiscriminator as Discriminator
from modules.style_transfer import StyleTransfer
from modules.memory_bank import MemoryBankModule
from modules.uncertainty_estimation import UncertaintyEstimation
from modules.few_shot_adapter import AdaptableColorizer


class TestGuideNet(unittest.TestCase):
    """Тесты для компонента GuideNet (сеть советов по цветам)."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.img_size = 256
        self.input_channels = 1  # Входной канал для ч/б изображения
        self.advice_channels = 2  # Выходные каналы a и b для цветового совета
        self.feature_dim = 64
        self.num_layers = 6
        
        # Создаем модель для тестов
        self.model = GuideNet(
            input_channels=self.input_channels,
            advice_channels=self.advice_channels,
            feature_dim=self.feature_dim,
            num_layers=self.num_layers,
            use_attention=True
        ).to(self.device)
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_channels, self.img_size, self.img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Проверяем, что выходные данные имеют правильную форму и структуру
        self.assertIsInstance(output, dict)
        self.assertIn('color_advice', output)
        
        # Проверяем форму тензора советов по цветам
        color_advice = output['color_advice']
        self.assertEqual(color_advice.shape, (batch_size, self.advice_channels, self.img_size, self.img_size))
        
        # Проверяем, что выход содержит карты уверенности, если они включены
        if self.model.use_confidence:
            self.assertIn('confidence_map', output)
            confidence_map = output['confidence_map']
            self.assertEqual(confidence_map.shape, (batch_size, 1, self.img_size, self.img_size))
            
            # Проверяем, что значения карты уверенности находятся в диапазоне [0, 1]
            self.assertTrue(torch.all(confidence_map >= 0.0))
            self.assertTrue(torch.all(confidence_map <= 1.0))
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через модель."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_channels, self.img_size, self.img_size).to(self.device)
        target_advice = torch.randn(batch_size, self.advice_channels, self.img_size, self.img_size).to(self.device)
        
        # Создаем оптимизатор
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        color_advice = output['color_advice']
        
        # Вычисляем потерю
        loss = nn.MSELoss()(color_advice, target_advice)
        
        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        
        # Проверяем, что градиенты не None и не содержат NaN для всех параметров модели
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Градиент для {name} равен None")
                self.assertFalse(torch.isnan(param.grad).any().item(), f"Градиент для {name} содержит NaN")
                
    def test_reward_system(self):
        """Тестирование системы наград для GuideNet."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_channels, self.img_size, self.img_size).to(self.device)
        target_advice = torch.randn(batch_size, self.advice_channels, self.img_size, self.img_size).to(self.device)
        
        # Получаем советы по цветам
        output = self.model(input_tensor)
        color_advice = output['color_advice']
        
        # Проверяем функцию вычисления награды
        if hasattr(self.model, 'compute_reward'):
            reward = self.model.compute_reward(color_advice, target_advice)
            
            # Проверяем, что награда имеет правильную форму и типы
            self.assertEqual(reward.shape, (batch_size,))
            self.assertEqual(reward.dtype, torch.float32)
            
            # Проверяем, что награда находится в ожидаемом диапазоне
            # Для многих реализаций, награда должна быть в диапазоне [-1, 1] или [0, 1]
            self.assertTrue(torch.all(reward >= -1.0))
            self.assertTrue(torch.all(reward <= 1.0))


class TestDiscriminator(unittest.TestCase):
    """Тесты для компонента Discriminator (дискриминатор для GAN)."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.img_size = 256
        self.input_nc = 3  # RGB изображение
        self.ndf = 64  # Число фильтров дискриминатора
        self.n_layers = 3
        
        # Создаем модель для тестов
        self.model = Discriminator(
            input_nc=self.input_nc,
            ndf=self.ndf,
            n_layers=self.n_layers,
            use_spectral_norm=True,
            reward_type='adaptive'
        ).to(self.device)
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_nc, self.img_size, self.img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Для PatchGAN дискриминатора ожидаем выход формы [batch_size, 1, patch_size, patch_size]
        # где patch_size зависит от количества слоев
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)
        
        # PatchGAN уменьшает размер изображения в 2^n_layers раз
        expected_patch_size = self.img_size // (2 ** self.n_layers)
        self.assertEqual(output.shape[2], expected_patch_size)
        self.assertEqual(output.shape[3], expected_patch_size)
    
    def test_gradient_flow(self):
        """Тестирование потока градиентов через модель."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.input_nc, self.img_size, self.img_size).to(self.device)
        
        # Создаем целевые значения (1 для реальных изображений, 0 для поддельных)
        target_real = torch.ones(batch_size, 1, self.img_size // (2 ** self.n_layers), 
                                 self.img_size // (2 ** self.n_layers)).to(self.device)
        
        # Создаем оптимизатор
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Вычисляем потерю (используем binary cross entropy)
        loss = nn.BCEWithLogitsLoss()(output, target_real)
        
        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        
        # Проверяем, что градиенты не None и не содержат NaN для всех параметров модели
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Градиент для {name} равен None")
                self.assertFalse(torch.isnan(param.grad).any().item(), f"Градиент для {name} содержит NaN")
                
    def test_compute_rewards(self):
        """Тестирование функции вычисления наград для генератора."""
        batch_size = 2
        input_fake = torch.randn(batch_size, self.input_nc, self.img_size, self.img_size).to(self.device)
        input_real = torch.randn(batch_size, self.input_nc, self.img_size, self.img_size).to(self.device)
        
        # Проверяем, что функция вычисления наград доступна
        self.assertTrue(hasattr(self.model, 'compute_rewards'))
        
        # Вычисляем награды
        rewards = self.model.compute_rewards(input_fake, input_real)
        
        # Проверяем форму и тип выхода
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(rewards.dtype, torch.float32)
        
        # Проверяем диапазон наград
        self.assertTrue(torch.all(rewards >= -1.0))
        self.assertTrue(torch.all(rewards <= 1.0))


class TestStyleTransfer(unittest.TestCase):
    """Тесты для компонента StyleTransfer."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.content_weight = 1.0
        self.style_weight = 1000.0
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        # Создаем модель для тестов
        self.model = StyleTransfer(
            content_weight=self.content_weight,
            style_weight=self.style_weight,
            content_layers=self.content_layers,
            style_layers=self.style_layers,
            device=self.device
        )
        
    def test_transfer_style(self):
        """Тестирование переноса стиля."""
        # Создаем контентное и стилевое изображения
        img_size = 128  # Используем меньшее разрешение для ускорения тестов
        content_image = torch.randn(1, 3, img_size, img_size).to(self.device)
        style_image = torch.randn(1, 3, img_size, img_size).to(self.device)
        
        # Выполняем перенос стиля с ограниченным числом итераций
        result = self.model.transfer_style(
            content_image=content_image, 
            style_image=style_image, 
            num_steps=2,  # Используем малое число итераций для быстроты тестов
            init_with_content=True
        )
        
        # Проверяем форму результата
        self.assertEqual(result.shape, content_image.shape)
        
    def test_content_style_losses(self):
        """Тестирование функций потери контента и стиля."""
        # Создаем контентное и стилевое изображения
        img_size = 128
        content_image = torch.randn(1, 3, img_size, img_size).to(self.device)
        style_image = torch.randn(1, 3, img_size, img_size).to(self.device)
        input_image = content_image.clone()  # Начинаем с контентного изображения
        
        # Проверяем, что функции вычисления потерь доступны
        self.assertTrue(hasattr(self.model, 'content_loss'))
        self.assertTrue(hasattr(self.model, 'style_loss'))
        
        # Вычисляем потери
        content_loss = self.model.content_loss(input_image, content_image)
        style_loss = self.model.style_loss(input_image, style_image)
        
        # Проверяем форму и тип выхода
        self.assertTrue(isinstance(content_loss, torch.Tensor))
        self.assertTrue(isinstance(style_loss, torch.Tensor))
        self.assertEqual(content_loss.shape, torch.Size([]))  # Скаляр
        self.assertEqual(style_loss.shape, torch.Size([]))  # Скаляр
        
        # Проверяем градиенты
        total_loss = content_loss + style_loss
        total_loss.backward()
        
        # Проверяем, что градиенты вычислены для входного изображения
        self.assertIsNotNone(input_image.grad)
        self.assertFalse(torch.isnan(input_image.grad).any().item())


class TestMemoryBank(unittest.TestCase):
    """Тесты для компонента MemoryBank."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.feature_dim = 256
        self.max_items = 100
        
        # Создаем модель для тестов
        self.model = MemoryBankModule(
            feature_dim=self.feature_dim,
            max_items=self.max_items,
            index_type='flat',
            use_fusion=True,
            device=self.device
        )
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        img_size = 128
        input_tensor = torch.randn(batch_size, 1, img_size, img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Проверяем структуру выходных данных
        self.assertIsInstance(output, dict)
        
        # Если банк памяти пуст, должен быть возвращен None или пустой результат
        if len(self.model.memory_items) == 0:
            self.assertFalse('colorized' in output or 
                             output.get('colorized') is not None and output['colorized'].numel() > 0)
        
    def test_add_item(self):
        """Тестирование добавления элементов в банк памяти."""
        # Создаем тестовые данные
        img_size = 64
        grayscale_img = torch.randn(1, 1, img_size, img_size).to(self.device)
        color_img = torch.randn(1, 3, img_size, img_size).to(self.device)
        
        # Проверяем, что функция добавления элемента доступна
        self.assertTrue(hasattr(self.model, 'add_item'))
        
        # Добавляем элемент в банк памяти
        self.model.add_item(grayscale_img, color_img)
        
        # Проверяем, что элемент был добавлен
        self.assertEqual(len(self.model.memory_items), 1)
        
        # Добавляем еще несколько элементов
        for _ in range(5):
            grayscale = torch.randn(1, 1, img_size, img_size).to(self.device)
            color = torch.randn(1, 3, img_size, img_size).to(self.device)
            self.model.add_item(grayscale, color)
            
        # Проверяем, что все элементы были добавлены
        self.assertEqual(len(self.model.memory_items), 6)
        
        # Проверяем ограничение на максимальное количество элементов
        # Добавляем элементы до превышения лимита
        for _ in range(self.max_items):
            grayscale = torch.randn(1, 1, img_size, img_size).to(self.device)
            color = torch.randn(1, 3, img_size, img_size).to(self.device)
            self.model.add_item(grayscale, color)
            
        # Проверяем, что количество элементов не превышает максимум
        self.assertLessEqual(len(self.model.memory_items), self.max_items)
        
    def test_query(self):
        """Тестирование запроса к банку памяти."""
        # Добавляем несколько элементов в банк памяти
        img_size = 64
        for _ in range(10):
            grayscale = torch.randn(1, 1, img_size, img_size).to(self.device)
            color = torch.randn(1, 3, img_size, img_size).to(self.device)
            self.model.add_item(grayscale, color)
            
        # Проверяем, что функция запроса доступна
        self.assertTrue(hasattr(self.model, 'query'))
        
        # Выполняем запрос
        query_img = torch.randn(1, 1, img_size, img_size).to(self.device)
        results = self.model.query(query_img, k=3)
        
        # Проверяем структуру результатов
        self.assertIsInstance(results, dict)
        self.assertIn('items', results)
        self.assertIn('distances', results)
        
        # Проверяем, что количество результатов соответствует запрошенному
        self.assertEqual(len(results['items']), min(3, len(self.model.memory_items)))


class TestUncertaintyEstimation(unittest.TestCase):
    """Тесты для компонента UncertaintyEstimation."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.method = 'guided'
        self.num_samples = 5
        self.dropout_rate = 0.2
        
        # Создаем модель для тестов
        self.model = UncertaintyEstimation(
            method=self.method,
            num_samples=self.num_samples,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        img_size = 128
        channels = 3
        input_tensor = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.model(input_tensor)
        
        # Проверяем структуру выходных данных
        self.assertIsInstance(output, dict)
        self.assertIn('uncertainty', output)
        
        # Проверяем форму карты неопределенности
        uncertainty_map = output['uncertainty']
        self.assertEqual(uncertainty_map.shape, (batch_size, 1, img_size, img_size))
        
        # Проверяем, что значения неопределенности неотрицательны
        self.assertTrue(torch.all(uncertainty_map >= 0.0))
    
    def test_monte_carlo_dropout(self):
        """Тестирование метода Monte Carlo Dropout для оценки неопределенности."""
        if self.model.method != 'mc_dropout':
            self.skipTest("Тест применим только для метода mc_dropout")
            
        batch_size = 2
        img_size = 64
        channels = 3
        input_tensor = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
        
        # Проверяем, что функция Monte Carlo Dropout доступна
        self.assertTrue(hasattr(self.model, 'monte_carlo_dropout'))
        
        # Вычисляем неопределенность методом Monte Carlo Dropout
        uncertainty = self.model.monte_carlo_dropout(input_tensor)
        
        # Проверяем форму выходных данных
        self.assertEqual(uncertainty.shape, (batch_size, 1, img_size, img_size))
        
        # Проверяем, что значения неопределенности неотрицательны
        self.assertTrue(torch.all(uncertainty >= 0.0))
        
    def test_ensembling_uncertainty(self):
        """Тестирование метода ансамблирования для оценки неопределенности."""
        if self.model.method != 'ensemble':
            self.skipTest("Тест применим только для метода ensemble")
            
        batch_size = 2
        img_size = 64
        channels = 3
        
        # Создаем набор предсказаний от разных моделей
        predictions = []
        for _ in range(5):  # Имитируем 5 моделей
            pred = torch.randn(batch_size, channels, img_size, img_size).to(self.device)
            predictions.append(pred)
            
        # Проверяем, что функция ансамблирования доступна
        self.assertTrue(hasattr(self.model, 'ensemble_uncertainty'))
        
        # Вычисляем неопределенность методом ансамблирования
        uncertainty = self.model.ensemble_uncertainty(predictions)
        
        # Проверяем форму выходных данных
        self.assertEqual(uncertainty.shape, (batch_size, 1, img_size, img_size))
        
        # Проверяем, что значения неопределенности неотрицательны
        self.assertTrue(torch.all(uncertainty >= 0.0))


class TestAdaptableColorizer(unittest.TestCase):
    """Тесты для компонента AdaptableColorizer (Few-shot/Cross-domain адаптер)."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Базовая конфигурация для тестирования
        self.adapter_type = 'standard'
        self.bottleneck_dim = 64
        
        # Создаем простую базовую модель для тестов
        class SimpleBaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 2, kernel_size=3, padding=1)
                )
                
            def forward(self, x):
                features = self.encoder(x)
                output = self.decoder(features)
                return {'a': output[:, 0:1], 'b': output[:, 1:2]}
        
        self.base_model = SimpleBaseModel().to(self.device)
        
        # Создаем адаптер для тестов
        self.adapter = AdaptableColorizer(
            adapter_type=self.adapter_type,
            bottleneck_dim=self.bottleneck_dim,
            base_model=self.base_model,
            device=self.device
        )
        
    def test_forward(self):
        """Тестирование прямого прохода."""
        batch_size = 2
        img_size = 64
        input_tensor = torch.randn(batch_size, 1, img_size, img_size).to(self.device)
        
        # Выполняем прямой проход
        output = self.adapter(input_tensor)
        
        # Проверяем структуру выходных данных
        self.assertIsInstance(output, dict)
        self.assertIn('a', output)
        self.assertIn('b', output)
        
        # Проверяем форму выходных данных
        self.assertEqual(output['a'].shape, (batch_size, 1, img_size, img_size))
        self.assertEqual(output['b'].shape, (batch_size, 1, img_size, img_size))
    
    def test_adaptation(self):
        """Тестирование адаптации модели на новый домен."""
        batch_size = 2
        img_size = 64
        
        # Создаем несколько образцов для адаптации
        source_grayscale = torch.randn(batch_size, 1, img_size, img_size).to(self.device)
        source_color_ab = torch.randn(batch_size, 2, img_size, img_size).to(self.device)
        
        # Проверяем, что функция адаптации доступна
        self.assertTrue(hasattr(self.adapter, 'adapt'))
        
        # Адаптируем модель
        self.adapter.adapt(source_grayscale, source_color_ab, num_iterations=2)
        
        # Проверяем, что параметры адаптера были обновлены
        # (это косвенно проверяется через наличие градиентов)
        for name, param in self.adapter.named_parameters():
            if 'adapter' in name and param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_reset_adaptation(self):
        """Тестирование сброса адаптации."""
        # Проверяем, что функция сброса адаптации доступна
        self.assertTrue(hasattr(self.adapter, 'reset_adaptation'))
        
        # Сбрасываем адаптацию
        self.adapter.reset_adaptation()
        
        # Сложно проверить результат сброса напрямую, но можно убедиться,
        # что модель все еще работает после сброса
        batch_size = 2
        img_size = 64
        input_tensor = torch.randn(batch_size, 1, img_size, img_size).to(self.device)
        
        # Выполняем прямой проход после сброса
        output = self.adapter(input_tensor)
        
        # Проверяем структуру выходных данных
        self.assertIsInstance(output, dict)
        self.assertIn('a', output)
        self.assertIn('b', output)


class TestMultipleModulesIntegration(unittest.TestCase):
    """Тесты для интеграции нескольких интеллектуальных модулей."""
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Создаем компоненты для тестов
        self.img_size = 128  # Используем меньшее разрешение для тестов
        
        # GuideNet
        self.guide_net = GuideNet(
            input_channels=1,
            advice_channels=2,
            feature_dim=64,
            num_layers=4,
            use_attention=True,
            device=self.device
        ).to(self.device)
        
        # Discriminator
        self.discriminator = Discriminator(
            input_nc=3,
            ndf=64,
            n_layers=2,
            use_spectral_norm=True,
            reward_type='adaptive',
            device=self.device
        ).to(self.device)
        
        # UncertaintyEstimation
        self.uncertainty = UncertaintyEstimation(
            method='guided',
            num_samples=3,
            dropout_rate=0.1,
            device=self.device
        ).to(self.device)
        
    def test_guide_discriminator_integration(self):
        """Тестирование интеграции GuideNet и Discriminator."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, self.img_size, self.img_size).to(self.device)
        
        # Получаем советы от GuideNet
        guide_output = self.guide_net(input_tensor)
        color_advice = guide_output['color_advice']
        
        # Преобразуем L-канал и советы a, b в RGB для передачи в дискриминатор
        # (Здесь для простоты просто предполагаем, что у нас уже RGB)
        fake_rgb = torch.cat([input_tensor, color_advice], dim=1)
        
        # Передаем результат в дискриминатор
        disc_output = self.discriminator(fake_rgb)
        
        # Проверяем форму выхода дискриминатора
        expected_patch_size = self.img_size // (2 ** 2)  # n_layers=2
        self.assertEqual(disc_output.shape, (batch_size, 1, expected_patch_size, expected_patch_size))
        
        # Проверяем вычисление наград
        rewards = self.discriminator.compute_rewards(fake_rgb, fake_rgb)
        
        # Проверяем форму наград
        self.assertEqual(rewards.shape, (batch_size,))
        
    def test_guidenet_uncertainty_integration(self):
        """Тестирование интеграции GuideNet и UncertaintyEstimation."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, self.img_size, self.img_size).to(self.device)
        
        # Получаем советы от GuideNet
        guide_output = self.guide_net(input_tensor)
        color_advice = guide_output['color_advice']
        
        # Оцениваем неопределенность советов
        uncertainty_output = self.uncertainty(color_advice)
        uncertainty_map = uncertainty_output['uncertainty']
        
        # Проверяем форму карты неопределенности
        self.assertEqual(uncertainty_map.shape, (batch_size, 1, self.img_size, self.img_size))
        
        # Проверяем, что карта неопределенности содержит разумные значения
        self.assertTrue(torch.all(uncertainty_map >= 0.0))
        self.assertTrue(torch.all(uncertainty_map <= 1.0))
        
        # Простая проверка: неопределенность не должна быть везде одинаковой
        # (это не всегда верно, но в большинстве случаев должно выполняться)
        for i in range(batch_size):
            # Проверяем, что есть хотя бы небольшая вариация в карте неопределенности
            std = torch.std(uncertainty_map[i])
            self.assertGreater(std, 0.0)


if __name__ == '__main__':
    unittest.main()