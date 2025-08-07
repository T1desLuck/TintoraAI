"""
Metrics: Модуль для оценки качества колоризации изображений.

Данный модуль предоставляет функции и классы для оценки качества результатов колоризации
с использованием различных метрик, включая PSNR, SSIM, LPIPS, колоритность и другие.
Эти метрики помогают объективно оценить качество колоризации и сравнить различные методы.

Ключевые особенности:
- Полный набор стандартных метрик для оценки качества изображений
- Специализированные метрики для оценки колоризации
- Метрики на основе перцептивного сходства
- Статистические инструменты для анализа результатов
- Интеграция с популярными библиотеками для оценки качества изображений

Преимущества для колоризации:
- Объективная оценка качества колоризации
- Оптимизация моделей на основе наиболее релевантных метрик
- Возможность сравнения различных методов и подходов
- Отслеживание прогресса в процессе обучения
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Union, Optional, Any
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.color import rgb2lab, lab2rgb
import cv2


class PSNR:
    """
    Peak Signal-to-Noise Ratio (PSNR) для оценки качества изображений.
    Высокие значения PSNR указывают на лучшее качество.
    
    Args:
        max_value (float): Максимальное значение пикселя
    """
    def __init__(self, max_value: float = 1.0):
        self.max_value = max_value
    
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет PSNR между двумя изображениями.
        
        Args:
            img1 (torch.Tensor): Первое изображение [B, C, H, W]
            img2 (torch.Tensor): Второе изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Значение PSNR
        """
        # Проверяем размеры
        if img1.shape != img2.shape:
            raise ValueError(f"Входные тензоры должны иметь одинаковые размеры, получены {img1.shape} и {img2.shape}")
        
        # Вычисляем MSE
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        
        # Предотвращаем деление на ноль
        mse = torch.clamp(mse, min=1e-10)
        
        # Вычисляем PSNR
        psnr_value = 20 * torch.log10(self.max_value / torch.sqrt(mse))
        
        return psnr_value
    
    @staticmethod
    def compute(img1: np.ndarray, img2: np.ndarray, max_value: float = 1.0) -> float:
        """
        Статический метод для вычисления PSNR между двумя изображениями.
        
        Args:
            img1 (np.ndarray): Первое изображение [H, W, C]
            img2 (np.ndarray): Второе изображение [H, W, C]
            max_value (float): Максимальное значение пикселя
            
        Returns:
            float: Значение PSNR
        """
        # Проверяем размеры
        if img1.shape != img2.shape:
            raise ValueError(f"Входные массивы должны иметь одинаковые размеры, получены {img1.shape} и {img2.shape}")
        
        return psnr_skimage(img1, img2, data_range=max_value)


class SSIM:
    """
    Structural Similarity Index (SSIM) для оценки качества изображений.
    Высокие значения SSIM указывают на лучшее структурное сходство.
    
    Args:
        window_size (int): Размер окна для вычисления SSIM
        channel_dim (int): Индекс измерения канала
    """
    def __init__(self, window_size: int = 11, channel_dim: int = 1):
        self.window_size = window_size
        self.channel_dim = channel_dim
        
        # Создаем ядро Гаусса для взвешивания
        self.gaussian_kernel = self._create_gaussian_kernel(window_size)
    
    def _create_gaussian_kernel(self, window_size: int) -> torch.Tensor:
        """
        Создает ядро Гаусса для вычисления SSIM.
        
        Args:
            window_size (int): Размер окна
            
        Returns:
            torch.Tensor: Ядро Гаусса
        """
        # Создаем 1D ядро Гаусса
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Создаем 2D ядро Гаусса
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        return kernel_2d.unsqueeze(0).unsqueeze(0)
    
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет SSIM между двумя изображениями.
        
        Args:
            img1 (torch.Tensor): Первое изображение [B, C, H, W]
            img2 (torch.Tensor): Второе изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Значение SSIM
        """
        # Проверяем размеры
        if img1.shape != img2.shape:
            raise ValueError(f"Входные тензоры должны иметь одинаковые размеры, получены {img1.shape} и {img2.shape}")
        
        # Получаем устройство для вычислений
        device = img1.device
        gaussian_kernel = self.gaussian_kernel.to(device)
        
        # Константы для стабильности
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Вычисляем средние значения
        mu1 = F.conv2d(img1, gaussian_kernel, padding=self.window_size // 2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, gaussian_kernel, padding=self.window_size // 2, groups=img2.shape[1])
        
        # Вычисляем квадраты средних значений
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Вычисляем дисперсии и ковариацию
        sigma1_sq = F.conv2d(img1 ** 2, gaussian_kernel, padding=self.window_size // 2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, gaussian_kernel, padding=self.window_size // 2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, gaussian_kernel, padding=self.window_size // 2, groups=img1.shape[1]) - mu1_mu2
        
        # Вычисляем SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        # Возвращаем среднее значение SSIM
        return torch.mean(ssim_map, dim=(1, 2, 3))
    
    @staticmethod
    def compute(img1: np.ndarray, img2: np.ndarray, multichannel: bool = True) -> float:
        """
        Статический метод для вычисления SSIM между двумя изображениями.
        
        Args:
            img1 (np.ndarray): Первое изображение [H, W, C] или [H, W]
            img2 (np.ndarray): Второе изображение [H, W, C] или [H, W]
            multichannel (bool): Учитывать ли каналы отдельно
            
        Returns:
            float: Значение SSIM
        """
        # Проверяем размеры
        if img1.shape != img2.shape:
            raise ValueError(f"Входные массивы должны иметь одинаковые размеры, получены {img1.shape} и {img2.shape}")
        
        return ssim_skimage(img1, img2, multichannel=multichannel, data_range=1.0)


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) для оценки перцептивного сходства.
    
    Эта метрика использует признаки из предобученной нейронной сети для оценки
    перцептивного сходства между изображениями. Низкие значения LPIPS указывают на лучшее сходство.
    
    Args:
        net_type (str): Тип базовой сети ('alex', 'vgg', 'squeeze')
        version (str): Версия LPIPS ('0.1')
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        super(LPIPS, self).__init__()
        
        # Проверяем тип сети
        if net_type not in ['alex', 'vgg', 'squeeze']:
            raise ValueError(f"Неподдерживаемый тип сети: {net_type}")
        
        # Загружаем базовую сеть
        if net_type == 'alex':
            self.net = models.alexnet(pretrained=True)
            self.features = self.net.features[:12]
            self.channels = [64, 192, 384, 256, 256]
        elif net_type == 'vgg':
            self.net = models.vgg16(pretrained=True)
            self.features = self.net.features[:30]
            self.channels = [64, 128, 256, 512, 512]
        else:  # squeeze
            self.net = models.squeezenet1_1(pretrained=True)
            self.features = nn.Sequential(*list(self.net.features.children())[:12])
            self.channels = [64, 128, 256, 384, 384]
        
        # Замораживаем параметры базовой сети
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Создаем линейные слои для взвешивания признаков
        self.lin_layers = nn.ModuleList([
            nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0, bias=False)
            for ch in self.channels
        ])
        
        # Инициализируем веса линейных слоев
        for i, layer in enumerate(self.lin_layers):
            layer.weight.data.normal_(0, 0.01)
            
        # Создаем дополнительные слои для нормализации и масштабирования
        self.scaling_layers = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
            for _ in self.channels
        ])
        
        # Инициализируем веса масштабирующих слоев
        for i, layer in enumerate(self.scaling_layers):
            layer.weight.data.fill_(1.0)
            
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет LPIPS между двумя изображениями.
        
        Args:
            x (torch.Tensor): Первое изображение [B, 3, H, W]
            y (torch.Tensor): Второе изображение [B, 3, H, W]
            
        Returns:
            torch.Tensor: Значение LPIPS
        """
        # Проверяем размеры
        if x.shape != y.shape:
            raise ValueError(f"Входные тензоры должны иметь одинаковые размеры, получены {x.shape} и {y.shape}")
        
        # Нормализуем входные изображения
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Извлекаем признаки
        x_features = []
        y_features = []
        
        # Проходим по слоям и извлекаем признаки
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            
            if i in [2, 5, 8, 10, 12]:  # Индексы слоев для извлечения признаков
                x_features.append(x)
                y_features.append(y)
        
        # Вычисляем расстояние между признаками
        diffs = []
        
        for i, (x_feat, y_feat) in enumerate(zip(x_features, y_features)):
            # Нормализуем признаки
            x_feat = x_feat / (torch.sqrt(torch.sum(x_feat ** 2, dim=1, keepdim=True)) + 1e-8)
            y_feat = y_feat / (torch.sqrt(torch.sum(y_feat ** 2, dim=1, keepdim=True)) + 1e-8)
            
            # Вычисляем разницу
            diff = (x_feat - y_feat) ** 2
            
            # Применяем линейный слой
            diff = self.lin_layers[i](diff)
            
            # Применяем масштабирование
            diff = self.scaling_layers[i](diff)
            
            # Усредняем по пространственным измерениям
            diff = torch.mean(diff, dim=(2, 3), keepdim=True)
            
            diffs.append(diff)
        
        # Суммируем расстояния между признаками
        lpips_value = torch.sum(torch.cat(diffs, dim=1), dim=1)
        
        return lpips_value


class Colorfulness:
    """
    Метрика колоритности изображения.
    
    Эта метрика оценивает, насколько "красочным" является изображение.
    Высокие значения указывают на более колоритное изображение.
    
    Основана на работе Hasler and Süsstrunk (2003): https://infoscience.epfl.ch/record/33994
    """
    @staticmethod
    def compute(image: np.ndarray) -> float:
        """
        Вычисляет колоритность изображения.
        
        Args:
            image (np.ndarray): Изображение RGB [H, W, 3] со значениями в диапазоне [0, 1] или [0, 255]
            
        Returns:
            float: Значение колоритности
        """
        # Нормализуем изображение, если нужно
        if image.max() <= 1.0:
            image = image * 255
        
        # Разделяем на каналы
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        
        # Вычисляем rg и yb компоненты
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # Вычисляем среднее и стандартное отклонение
        rg_mean = np.mean(rg)
        rg_std = np.std(rg)
        yb_mean = np.mean(yb)
        yb_std = np.std(yb)
        
        # Вычисляем колоритность
        rg_yb_mean = np.sqrt(rg_mean ** 2 + yb_mean ** 2)
        rg_yb_std = np.sqrt(rg_std ** 2 + yb_std ** 2)
        
        colorfulness = rg_yb_std + 0.3 * rg_yb_mean
        
        return colorfulness
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет колоритность изображения.
        
        Args:
            image (torch.Tensor): Изображение RGB [B, 3, H, W] со значениями в диапазоне [0, 1]
            
        Returns:
            torch.Tensor: Значение колоритности
        """
        # Проверяем размеры
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"Входной тензор должен иметь форму [B, 3, H, W], получен {image.shape}")
        
        # Переводим в формат [B, H, W, 3]
        image = image.permute(0, 2, 3, 1)
        
        # Разделяем на каналы
        R = image[:, :, :, 0]
        G = image[:, :, :, 1]
        B = image[:, :, :, 2]
        
        # Вычисляем rg и yb компоненты
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # Вычисляем среднее и стандартное отклонение для каждого изображения в пакете
        rg_mean = torch.mean(rg, dim=(1, 2))
        rg_std = torch.std(rg, dim=(1, 2))
        yb_mean = torch.mean(yb, dim=(1, 2))
        yb_std = torch.std(yb, dim=(1, 2))
        
        # Вычисляем колоритность
        rg_yb_mean = torch.sqrt(rg_mean ** 2 + yb_mean ** 2)
        rg_yb_std = torch.sqrt(rg_std ** 2 + yb_std ** 2)
        
        colorfulness = rg_yb_std + 0.3 * rg_yb_mean
        
        return colorfulness


class FID:
    """
    Fréchet Inception Distance (FID) для оценки качества генеративных моделей.
    
    FID измеряет расстояние между распределениями признаков реальных и сгенерированных изображений,
    извлеченных с помощью Inception v3. Низкие значения FID указывают на лучшее качество.
    
    Примечание: Для полной функциональности требуется библиотека scipy.
    """
    def __init__(self, dims: int = 2048, device: str = 'cpu'):
        self.dims = dims
        self.device = device
        
        # Загружаем Inception v3
        try:
            self.inception = models.inception_v3(pretrained=True, transform_input=False)
            self.inception.fc = nn.Identity()  # Убираем последний слой
            self.inception.eval()
            self.inception.to(device)
        except Exception as e:
            print(f"Не удалось загрузить Inception v3: {e}")
            self.inception = None
    
    def _calculate_activation_statistics(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет среднее значение и ковариационную матрицу активаций Inception.
        
        Args:
            images (torch.Tensor): Пакет изображений [B, 3, H, W] со значениями в диапазоне [0, 1]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Среднее значение и ковариационная матрица
        """
        # Проверяем, что Inception v3 загружен
        if self.inception is None:
            raise RuntimeError("Inception v3 не был загружен")
        
        # Преобразуем значения в диапазон [-1, 1]
        images = 2 * images - 1
        
        # Изменяем размер изображений, если нужно
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Извлекаем признаки
        with torch.no_grad():
            pred = self.inception(images)
            
        # Преобразуем в numpy
        pred = pred.cpu().numpy()
        
        # Вычисляем среднее значение и ковариационную матрицу
        mu = np.mean(pred, axis=0)
        sigma = np.cov(pred, rowvar=False)
        
        return mu, sigma
    
    def _calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """
        Вычисляет расстояние Фреше между двумя многомерными нормальными распределениями.
        
        Args:
            mu1 (np.ndarray): Среднее значение первого распределения
            sigma1 (np.ndarray): Ковариационная матрица первого распределения
            mu2 (np.ndarray): Среднее значение второго распределения
            sigma2 (np.ndarray): Ковариационная матрица второго распределения
            
        Returns:
            float: Расстояние Фреше
        """
        try:
            from scipy import linalg
        except ImportError:
            raise ImportError("Для вычисления FID требуется библиотека scipy")
        
        # Вычисляем квадрат разницы средних значений
        diff = mu1 - mu2
        
        # Вычисляем произведение ковариационных матриц
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Проверяем на комплексные числа
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Вычисляем расстояние Фреше
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    def __call__(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """
        Вычисляет FID между наборами реальных и сгенерированных изображений.
        
        Args:
            real_images (torch.Tensor): Реальные изображения [B, 3, H, W]
            generated_images (torch.Tensor): Сгенерированные изображения [B, 3, H, W]
            
        Returns:
            float: Значение FID
        """
        # Вычисляем статистики активаций
        mu1, sigma1 = self._calculate_activation_statistics(real_images)
        mu2, sigma2 = self._calculate_activation_statistics(generated_images)
        
        # Вычисляем расстояние Фреше
        fid_value = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return fid_value


class LabColorAccuracy:
    """
    Метрика точности цветов в пространстве Lab.
    
    Эта метрика оценивает точность цветов в пространстве Lab, которое лучше соответствует
    восприятию человека, чем RGB. Сравнение происходит только для каналов a и b.
    
    Args:
        threshold (float): Порог для определения "правильного" цвета
    """
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет точность цветов между предсказанным и целевым изображениями.
        
        Args:
            pred (torch.Tensor): Предсказанное изображение [B, 3, H, W] в пространстве Lab
            target (torch.Tensor): Целевое изображение [B, 3, H, W] в пространстве Lab
            
        Returns:
            torch.Tensor: Точность цветов
        """
        # Проверяем размеры
        if pred.shape != target.shape:
            raise ValueError(f"Входные тензоры должны иметь одинаковые размеры, получены {pred.shape} и {target.shape}")
        
        # Выделяем каналы a и b
        pred_ab = pred[:, 1:, :, :]
        target_ab = target[:, 1:, :, :]
        
        # Вычисляем евклидово расстояние
        dist = torch.sqrt(torch.sum((pred_ab - target_ab) ** 2, dim=1))
        
        # Определяем правильно предсказанные пиксели
        correct = dist < self.threshold
        
        # Вычисляем точность для каждого изображения
        accuracy = torch.mean(correct.float(), dim=(1, 2))
        
        return accuracy
    
    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray, threshold: float = 0.2) -> float:
        """
        Статический метод для вычисления точности цветов.
        
        Args:
            pred (np.ndarray): Предсказанное изображение [H, W, 3] в пространстве Lab
            target (np.ndarray): Целевое изображение [H, W, 3] в пространстве Lab
            threshold (float): Порог для определения "правильного" цвета
            
        Returns:
            float: Точность цветов
        """
        # Проверяем размеры
        if pred.shape != target.shape:
            raise ValueError(f"Входные массивы должны иметь одинаковые размеры, получены {pred.shape} и {target.shape}")
        
        # Выделяем каналы a и b
        pred_ab = pred[:, :, 1:]
        target_ab = target[:, :, 1:]
        
        # Вычисляем евклидово расстояние
        dist = np.sqrt(np.sum((pred_ab - target_ab) ** 2, axis=2))
        
        # Определяем правильно предсказанные пиксели
        correct = dist < threshold
        
        # Вычисляем точность
        accuracy = np.mean(correct)
        
        return accuracy


class ColorConsistency:
    """
    Метрика согласованности цветов для оценки стабильности колоризации.
    
    Эта метрика оценивает, насколько согласованно колоризуются схожие объекты или регионы
    на изображении. Высокие значения указывают на лучшую согласованность.
    
    Args:
        segmentation_model (nn.Module): Модель сегментации для выделения объектов (опционально)
        device (str): Устройство для вычислений
    """
    def __init__(self, segmentation_model: Optional[nn.Module] = None, device: str = 'cpu'):
        self.segmentation_model = segmentation_model
        self.device = device
    
    def __call__(self, colorized: torch.Tensor, segments: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Вычисляет согласованность цветов.
        
        Args:
            colorized (torch.Tensor): Колоризованное изображение [B, 3, H, W]
            segments (torch.Tensor, optional): Сегментация изображения [B, 1, H, W]
            
        Returns:
            torch.Tensor: Согласованность цветов
        """
        # Если сегментация не предоставлена и модель сегментации доступна,
        # выполняем сегментацию
        if segments is None and self.segmentation_model is not None:
            with torch.no_grad():
                segments = self.segmentation_model(colorized)
        
        # Если сегментация все еще не доступна, используем простой метод
        if segments is None:
            # Разбиваем изображение на сетку
            grid_size = 16
            B, C, H, W = colorized.shape
            segments = torch.zeros((B, 1, H, W), device=colorized.device)
            
            for i in range(0, grid_size):
                for j in range(0, grid_size):
                    h_start = i * H // grid_size
                    h_end = (i + 1) * H // grid_size
                    w_start = j * W // grid_size
                    w_end = (j + 1) * W // grid_size
                    
                    segments[:, :, h_start:h_end, w_start:w_end] = i * grid_size + j
        
        # Конвертируем в Lab пространство для лучшей оценки цветов
        colorized_ab = colorized[:, 1:, :, :]
        
        # Получаем уникальные сегменты
        batch_size = colorized.shape[0]
        consistency = []
        
        for b in range(batch_size):
            # Получаем уникальные сегменты для текущего изображения
            unique_segments = torch.unique(segments[b])
            
            # Вычисляем вариацию цветов внутри каждого сегмента
            segment_variances = []
            
            for seg_id in unique_segments:
                # Создаем маску для текущего сегмента
                mask = (segments[b] == seg_id).float()
                
                # Если сегмент слишком маленький, пропускаем его
                if torch.sum(mask) < 10:
                    continue
                
                # Извлекаем цвета для текущего сегмента
                masked_colors = colorized_ab[b] * mask
                
                # Вычисляем среднее значение цвета
                segment_mean = torch.sum(masked_colors, dim=(1, 2)) / torch.sum(mask)
                
                # Вычисляем вариацию цвета
                segment_variance = torch.mean(torch.sqrt(torch.sum((colorized_ab[b] - segment_mean.view(2, 1, 1)) ** 2, dim=0)) * mask)
                
                segment_variances.append(segment_variance)
            
            # Вычисляем среднюю вариацию по всем сегментам
            if len(segment_variances) > 0:
                avg_variance = torch.mean(torch.stack(segment_variances))
                
                # Преобразуем в меру согласованности (высокие значения = лучше)
                segment_consistency = 1.0 / (1.0 + avg_variance)
                consistency.append(segment_consistency)
            else:
                # Если нет достаточно больших сегментов, присваиваем низкое значение согласованности
                consistency.append(torch.tensor(0.5, device=colorized.device))
        
        return torch.stack(consistency)


class MetricsCalculator:
    """
    Калькулятор метрик для оценки качества колоризации.
    
    Этот класс объединяет различные метрики и упрощает их вычисление и отслеживание.
    
    Args:
        metrics (List[str]): Список метрик для вычисления
        lpips_net_type (str): Тип сети для LPIPS ('alex', 'vgg', 'squeeze')
        device (str): Устройство для вычислений
    """
    def __init__(self, metrics: List[str] = None, lpips_net_type: str = 'alex', device: str = 'cpu'):
        self.device = device
        
        # Определяем метрики по умолчанию, если не указаны
        if metrics is None:
            metrics = ['psnr', 'ssim', 'lpips', 'colorfulness']
        
        self.metric_names = metrics
        self.metrics = {}
        
        # Инициализируем метрики
        if 'psnr' in metrics:
            self.metrics['psnr'] = PSNR()
        
        if 'ssim' in metrics:
            self.metrics['ssim'] = SSIM()
        
        if 'lpips' in metrics:
            try:
                self.metrics['lpips'] = LPIPS(net_type=lpips_net_type).to(device)
            except Exception as e:
                print(f"Не удалось инициализировать LPIPS: {e}")
                self.metrics['lpips'] = None
        
        if 'colorfulness' in metrics:
            self.metrics['colorfulness'] = Colorfulness()
        
        if 'lab_accuracy' in metrics:
            self.metrics['lab_accuracy'] = LabColorAccuracy()
        
        if 'consistency' in metrics:
            self.metrics['consistency'] = ColorConsistency(device=device)
            
        if 'fid' in metrics:
            try:
                self.metrics['fid'] = FID(device=device)
            except Exception as e:
                print(f"Не удалось инициализировать FID: {e}")
                self.metrics['fid'] = None
        
        # Словарь для хранения значений метрик
        self.values = {name: [] for name in metrics}
        
    def calculate(self, 
                 pred: torch.Tensor, 
                 target: Optional[torch.Tensor] = None, 
                 segments: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Вычисляет метрики для колоризованных изображений.
        
        Args:
            pred (torch.Tensor): Предсказанные (колоризованные) изображения [B, C, H, W]
            target (torch.Tensor, optional): Целевые (эталонные) изображения [B, C, H, W]
            segments (torch.Tensor, optional): Сегментация изображений [B, 1, H, W]
            
        Returns:
            Dict[str, float]: Словарь с вычисленными метриками
        """
        results = {}
        
        # Переводим тензоры на нужное устройство
        pred = pred.to(self.device)
        if target is not None:
            target = target.to(self.device)
        if segments is not None:
            segments = segments.to(self.device)
        
        # Вычисляем метрики
        for name, metric in self.metrics.items():
            if metric is None:
                continue
                
            try:
                if name in ['psnr', 'ssim', 'lpips', 'lab_accuracy'] and target is not None:
                    # Метрики, требующие эталонное изображение
                    value = metric(pred, target)
                elif name == 'colorfulness':
                    # Метрика, требующая только колоризованное изображение
                    value = metric(pred)
                elif name == 'consistency':
                    # Метрика, требующая колоризованное изображение и сегментацию
                    value = metric(pred, segments)
                elif name == 'fid' and target is not None:
                    # FID требует наборы изображений
                    value = metric(target, pred)
                else:
                    # Пропускаем метрики, для которых нет нужных данных
                    continue
                    
                # Если значение - тензор, преобразуем в среднее
                if isinstance(value, torch.Tensor):
                    value = value.mean().item()
                    
                results[name] = value
                self.values[name].append(value)
                
            except Exception as e:
                print(f"Ошибка при вычислении метрики {name}: {e}")
                results[name] = float('nan')
        
        return results
    
    def reset(self):
        """Сбрасывает сохраненные значения метрик."""
        self.values = {name: [] for name in self.metric_names}
    
    def get_average(self) -> Dict[str, float]:
        """
        Возвращает средние значения метрик.
        
        Returns:
            Dict[str, float]: Словарь со средними значениями метрик
        """
        results = {}
        
        for name, values in self.values.items():
            if len(values) > 0:
                results[name] = sum(values) / len(values)
            else:
                results[name] = float('nan')
                
        return results
    
    def print_metrics(self, prefix: str = ""):
        """
        Выводит средние значения метрик.
        
        Args:
            prefix (str): Префикс для вывода
        """
        averages = self.get_average()
        
        print(f"{prefix}Метрики:")
        for name, value in averages.items():
            print(f"{prefix}  {name}: {value:.4f}")


class MetricsLogger:
    """
    Логгер метрик для отслеживания изменений во времени.
    
    Args:
        log_dir (str): Директория для сохранения логов
        metrics (List[str]): Список метрик для отслеживания
    """
    def __init__(self, log_dir: str = "./logs", metrics: List[str] = None):
        self.log_dir = log_dir
        
        # Создаем директорию для логов, если нужно
        os.makedirs(log_dir, exist_ok=True)
        
        # Определяем метрики по умолчанию, если не указаны
        if metrics is None:
            metrics = ['psnr', 'ssim', 'lpips', 'colorfulness']
        
        self.metrics = metrics
        
        # Словарь для хранения значений метрик
        self.history = {name: [] for name in metrics}
        
        # Счетчик шагов
        self.steps = []
        
    def log(self, metrics_dict: Dict[str, float], step: int = None):
        """
        Логирует значения метрик.
        
        Args:
            metrics_dict (Dict[str, float]): Словарь с значениями метрик
            step (int, optional): Номер шага
        """
        # Определяем номер шага
        if step is None:
            step = len(self.steps) + 1
            
        self.steps.append(step)
        
        # Логируем значения метрик
        for name in self.metrics:
            if name in metrics_dict:
                self.history[name].append(metrics_dict[name])
            else:
                self.history[name].append(float('nan'))
                
    def save(self, filename: str = "metrics_log.csv"):
        """
        Сохраняет историю метрик в CSV файл.
        
        Args:
            filename (str): Имя файла
        """
        # Определяем путь для сохранения
        save_path = os.path.join(self.log_dir, filename)
        
        # Создаем заголовок
        header = "step," + ",".join(self.metrics)
        
        # Создаем строки с данными
        rows = []
        for i, step in enumerate(self.steps):
            row = [str(step)]
            for name in self.metrics:
                row.append(str(self.history[name][i]))
            rows.append(",".join(row))
            
        # Сохраняем в файл
        with open(save_path, 'w') as f:
            f.write(header + "\n")
            f.write("\n".join(rows))
            
        print(f"Метрики сохранены в {save_path}")
        
    def load(self, filename: str = "metrics_log.csv"):
        """
        Загружает историю метрик из CSV файла.
        
        Args:
            filename (str): Имя файла
        """
        # Определяем путь для загрузки
        load_path = os.path.join(self.log_dir, filename)
        
        # Проверяем существование файла
        if not os.path.isfile(load_path):
            print(f"Файл {load_path} не существует")
            return
            
        # Загружаем данные
        with open(load_path, 'r') as f:
            lines = f.readlines()
            
        # Проверяем формат
        if len(lines) < 1:
            print(f"Файл {load_path} пуст")
            return
            
        # Получаем заголовок
        header = lines[0].strip().split(',')
        
        # Проверяем соответствие метрик
        file_metrics = header[1:]
        if file_metrics != self.metrics:
            print(f"Предупреждение: метрики в файле ({file_metrics}) не соответствуют текущим метрикам ({self.metrics})")
            
        # Сбрасываем историю
        self.steps = []
        self.history = {name: [] for name in self.metrics}
        
        # Загружаем данные
        for line in lines[1:]:
            values = line.strip().split(',')
            
            # Проверяем формат
            if len(values) != len(header):
                print(f"Пропускаем строку с неверным форматом: {line}")
                continue
                
            # Добавляем шаг
            self.steps.append(int(values[0]))
            
            # Добавляем значения метрик
            for i, name in enumerate(file_metrics):
                if name in self.metrics:
                    try:
                        value = float(values[i+1])
                    except ValueError:
                        value = float('nan')
                    self.history[name].append(value)
                    
        print(f"Метрики загружены из {load_path}")
        
    def plot(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Строит графики изменения метрик.
        
        Args:
            save_path (str, optional): Путь для сохранения графика
            figsize (Tuple[int, int]): Размер фигуры
        """
        try:
            import matplotlib.pyplot as plt
            
            # Определяем количество подграфиков
            nrows = (len(self.metrics) + 1) // 2
            ncols = min(2, len(self.metrics))
            
            # Создаем фигуру
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            
            # Если только одна метрика, преобразуем оси
            if len(self.metrics) == 1:
                axes = np.array([axes])
            
            # Приводим оси к 2D массиву
            if nrows == 1 and ncols == 1:
                axes = axes.reshape(1, 1)
            elif nrows == 1 or ncols == 1:
                axes = axes.reshape(-1, 1) if ncols == 1 else axes.reshape(1, -1)
            
            # Строим графики
            for i, name in enumerate(self.metrics):
                row = i // ncols
                col = i % ncols
                
                axes[row, col].plot(self.steps, self.history[name])
                axes[row, col].set_title(name)
                axes[row, col].set_xlabel("Шаг")
                axes[row, col].set_ylabel("Значение")
                axes[row, col].grid(True)
                
            # Скрываем оси для пустых ячеек
            for i in range(len(self.metrics), nrows * ncols):
                row = i // ncols
                col = i % ncols
                axes[row, col].axis('off')
                
            # Настраиваем общие параметры
            plt.tight_layout()
            
            # Сохраняем или отображаем график
            if save_path:
                plt.savefig(save_path)
                print(f"График сохранен в {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Для построения графиков требуется библиотека matplotlib")


# Функция для создания калькулятора метрик
def create_metrics_calculator(metrics: List[str] = None, lpips_net_type: str = 'alex', device: str = 'cpu') -> MetricsCalculator:
    """
    Создает калькулятор метрик для оценки качества колоризации.
    
    Args:
        metrics (List[str]): Список метрик для вычисления
        lpips_net_type (str): Тип сети для LPIPS ('alex', 'vgg', 'squeeze')
        device (str): Устройство для вычислений
        
    Returns:
        MetricsCalculator: Калькулятор метрик
    """
    return MetricsCalculator(metrics=metrics, lpips_net_type=lpips_net_type, device=device)


if __name__ == "__main__":
    # Пример использования модуля метрик
    
    # Создаем тестовые данные
    batch_size = 2
    img_size = 64
    original = torch.rand(batch_size, 3, img_size, img_size)
    colorized = torch.rand(batch_size, 3, img_size, img_size)
    
    try:
        # Создаем калькулятор метрик
        metrics_calculator = create_metrics_calculator(
            metrics=['psnr', 'ssim', 'colorfulness'],
            device='cpu'
        )
        
        # Вычисляем метрики
        metrics_values = metrics_calculator.calculate(colorized, original)
        
        # Выводим результаты
        print("Метрики:")
        for name, value in metrics_values.items():
            print(f"  {name}: {value:.4f}")
        
    except Exception as e:
        print(f"Ошибка при вычислении метрик: {e}")