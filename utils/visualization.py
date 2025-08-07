"""
Visualization: Модуль для визуализации результатов колоризации.

Данный модуль предоставляет функции и классы для визуализации результатов колоризации,
включая сравнение оригинальных и колоризованных изображений, отображение карт неопределенности,
создание цветовых палитр и другие визуальные представления результатов работы системы.

Ключевые особенности:
- Создание сравнительных изображений "до/после" для оценки качества колоризации
- Визуализация карт неопределенности с использованием цветовых схем
- Генерация цветовых палитр на основе результатов колоризации
- Интерактивные графики для анализа результатов работы модели
- Экспорт визуализаций в различные форматы для отчетов и презентаций

Преимущества для колоризации:
- Наглядное представление результатов для оценки качества
- Выявление проблемных областей колоризации через визуализацию неопределенности
- Возможность создания пользовательских визуализаций для конкретных задач
- Интеграция с другими компонентами системы для комплексного анализа
"""

import os
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any
from torchvision.utils import make_grid
from sklearn.cluster import KMeans


class ColorizationVisualizer:
    """
    Визуализатор результатов колоризации.
    
    Args:
        output_dir (str): Директория для сохранения результатов
        create_dirs (bool): Создавать ли поддиректории для результатов
        dpi (int): Разрешение для сохранения изображений
    """
    def __init__(self, output_dir: str = "./output", create_dirs: bool = True, dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Создаем директории для результатов, если нужно
        if create_dirs:
            self._create_output_directories()
    
    def _create_output_directories(self):
        """Создает директории для результатов визуализации."""
        os.makedirs(os.path.join(self.output_dir, "colorized"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "uncertainty_maps"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "palettes"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "attention_maps"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metrics"), exist_ok=True)
    
    def create_comparison(self, 
                          grayscale: np.ndarray, 
                          colorized: np.ndarray, 
                          original: Optional[np.ndarray] = None,
                          uncertainty: Optional[np.ndarray] = None,
                          filename: Optional[str] = None,
                          show_grid: bool = True) -> Image.Image:
        """
        Создает сравнительное изображение результатов колоризации.
        
        Args:
            grayscale (np.ndarray): ЧБ изображение
            colorized (np.ndarray): Колоризованное изображение
            original (np.ndarray, optional): Оригинальное цветное изображение
            uncertainty (np.ndarray, optional): Карта неопределенности
            filename (str, optional): Имя файла для сохранения
            show_grid (bool): Показывать ли сетку между изображениями
            
        Returns:
            PIL.Image.Image: Сравнительное изображение
        """
        # Определяем количество изображений для отображения
        n_images = 2  # Минимум - ЧБ и колоризованное
        if original is not None:
            n_images += 1
        if uncertainty is not None:
            n_images += 1
            
        # Определяем размер сетки
        if n_images <= 2:
            nrows, ncols = 1, n_images
        else:
            nrows = 2
            ncols = math.ceil(n_images / 2)
            
        # Создаем фигуру
        fig = plt.figure(figsize=(ncols * 5, nrows * 5))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        
        # Отображаем ЧБ изображение
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(grayscale, cmap='gray')
        ax.set_title("Исходное ЧБ")
        ax.axis('off')
        
        # Отображаем колоризованное изображение
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(colorized)
        ax.set_title("Колоризованное")
        ax.axis('off')
        
        # Отображаем оригинальное изображение, если предоставлено
        if original is not None:
            ax = fig.add_subplot(gs[1, 0] if nrows > 1 else gs[0, 2])
            ax.imshow(original)
            ax.set_title("Оригинал")
            ax.axis('off')
        
        # Отображаем карту неопределенности, если предоставлена
        if uncertainty is not None:
            ax = fig.add_subplot(gs[1, 1] if nrows > 1 else gs[0, 3])
            cmap = plt.cm.viridis
            im = ax.imshow(uncertainty, cmap=cmap)
            ax.set_title("Карта неопределенности")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Настраиваем общие параметры
        plt.tight_layout()
        if not show_grid:
            plt.subplots_adjust(wspace=0, hspace=0)
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "comparisons", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Сравнение сохранено: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def create_uncertainty_map(self,
                              uncertainty: np.ndarray,
                              colorized: Optional[np.ndarray] = None,
                              alpha: float = 0.7,
                              colormap: str = 'viridis',
                              filename: Optional[str] = None) -> Image.Image:
        """
        Создает визуализацию карты неопределенности.
        
        Args:
            uncertainty (np.ndarray): Карта неопределенности
            colorized (np.ndarray, optional): Колоризованное изображение
            alpha (float): Прозрачность наложения
            colormap (str): Название цветовой карты
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Визуализация карты неопределенности
        """
        # Определяем размер фигуры
        fig_width = 10
        fig_height = 5 if colorized is not None else 5
        
        # Создаем фигуру
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        if colorized is not None:
            # Создаем сетку из 1 строки и 2 столбцов
            gs = gridspec.GridSpec(1, 2, figure=fig)
            
            # Отображаем колоризованное изображение
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(colorized)
            ax.set_title("Колоризованное изображение")
            ax.axis('off')
            
            # Отображаем карту неопределенности
            ax = fig.add_subplot(gs[0, 1])
        else:
            # Если колоризованное изображение не предоставлено, используем всю фигуру
            ax = fig.add_subplot(111)
        
        # Выбираем цветовую карту
        cmap = plt.get_cmap(colormap)
        
        # Отображаем карту неопределенности
        im = ax.imshow(uncertainty, cmap=cmap)
        ax.set_title("Карта неопределенности")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "uncertainty_maps", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Карта неопределенности сохранена: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def create_attention_map(self,
                            attention: np.ndarray,
                            image: np.ndarray,
                            alpha: float = 0.6,
                            colormap: str = 'hot',
                            filename: Optional[str] = None) -> Image.Image:
        """
        Создает визуализацию карты внимания.
        
        Args:
            attention (np.ndarray): Карта внимания
            image (np.ndarray): Изображение
            alpha (float): Прозрачность наложения
            colormap (str): Название цветовой карты
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Визуализация карты внимания
        """
        # Нормализуем карту внимания, если нужно
        if attention.max() > 1.0 or attention.min() < 0.0:
            attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        # Изменяем размер карты внимания, если нужно
        if attention.shape != image.shape[:2]:
            attention = cv2.resize(attention, (image.shape[1], image.shape[0]))
        
        # Создаем фигуру
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Отображаем изображение
        axes[0].imshow(image)
        axes[0].set_title("Изображение")
        axes[0].axis('off')
        
        # Отображаем карту внимания
        axes[1].imshow(attention, cmap=colormap)
        axes[1].set_title("Карта внимания")
        axes[1].axis('off')
        
        # Отображаем наложение
        axes[2].imshow(image)
        attention_colored = plt.cm.get_cmap(colormap)(attention)
        attention_colored = attention_colored[..., :3]  # Убираем альфа-канал
        axes[2].imshow(attention_colored, alpha=alpha)
        axes[2].set_title("Наложение")
        axes[2].axis('off')
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "attention_maps", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Карта внимания сохранена: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def create_color_palette(self,
                           image: np.ndarray,
                           n_colors: int = 8,
                           sort_by: str = 'prominence',
                           filename: Optional[str] = None) -> Tuple[Image.Image, np.ndarray]:
        """
        Создает цветовую палитру на основе изображения.
        
        Args:
            image (np.ndarray): Изображение
            n_colors (int): Количество цветов в палитре
            sort_by (str): Метод сортировки цветов ('prominence', 'hue', 'saturation')
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            Tuple[PIL.Image.Image, np.ndarray]: Изображение с палитрой и массив цветов
        """
        # Преобразуем в формат RGB, если нужно
        if image.ndim == 2:
            # Если изображение в оттенках серого, преобразуем в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # Если изображение с альфа-каналом, убираем его
            image_rgb = image[..., :3]
        else:
            image_rgb = image.copy()
        
        # Изменяем форму для кластеризации
        pixels = image_rgb.reshape(-1, 3)
        
        # Применяем кластеризацию для выделения основных цветов
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(np.uint8)
        counts = np.bincount(kmeans.labels_)
        
        # Сортируем цвета по выбранному методу
        if sort_by == 'prominence':
            # Сортировка по количеству пикселей каждого цвета
            idx = np.argsort(-counts)
            colors = colors[idx]
        elif sort_by == 'hue':
            # Сортировка по оттенку (H в HSV)
            hsv_colors = cv2.cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            idx = np.argsort(hsv_colors[:, 0])
            colors = colors[idx]
        elif sort_by == 'saturation':
            # Сортировка по насыщенности (S в HSV)
            hsv_colors = cv2.cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            idx = np.argsort(-hsv_colors[:, 1])
            colors = colors[idx]
        
        # Создаем изображение с палитрой
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Отображаем изображение
        axes[0].imshow(image_rgb)
        axes[0].set_title("Изображение")
        axes[0].axis('off')
        
        # Отображаем цветовую палитру
        palette_height = 100
        palette_width = 400
        palette_image = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        # Заполняем палитру цветами
        step = palette_width // n_colors
        for i in range(n_colors):
            start = i * step
            end = (i + 1) * step if i < n_colors - 1 else palette_width
            palette_image[:, start:end] = colors[i]
        
        # Отображаем палитру
        axes[1].imshow(palette_image)
        axes[1].set_title("Цветовая палитра")
        axes[1].axis('off')
        
        # Добавляем аннотации с RGB значениями
        for i in range(n_colors):
            color = colors[i]
            x = i * step + step // 2
            text_color = 'black' if np.mean(color) > 128 else 'white'
            axes[1].text(x, palette_height // 2, f"RGB: {color[0]},{color[1]},{color[2]}", 
                         ha='center', va='center', color=text_color, fontsize=8)
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "palettes", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Цветовая палитра сохранена: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img, colors
    
    def create_multiple_variants(self,
                                grayscale: np.ndarray,
                                variants: List[np.ndarray],
                                titles: Optional[List[str]] = None,
                                uncertainty: Optional[List[np.ndarray]] = None,
                                filename: Optional[str] = None) -> Image.Image:
        """
        Создает визуализацию нескольких вариантов колоризации.
        
        Args:
            grayscale (np.ndarray): ЧБ изображение
            variants (List[np.ndarray]): Список вариантов колоризации
            titles (List[str], optional): Список заголовков для вариантов
            uncertainty (List[np.ndarray], optional): Список карт неопределенности
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Визуализация вариантов колоризации
        """
        # Определяем количество вариантов
        n_variants = len(variants)
        
        # Генерируем заголовки, если не предоставлены
        if titles is None:
            titles = [f"Вариант {i+1}" for i in range(n_variants)]
        
        # Определяем размер сетки
        n_images = n_variants + 1  # +1 для ЧБ изображения
        if uncertainty is not None:
            n_images += n_variants  # Добавляем карты неопределенности
        
        # Определяем количество строк и столбцов
        if uncertainty is not None:
            nrows = 2
            ncols = n_variants + 1
        else:
            nrows = math.ceil(n_images / 4)
            ncols = min(4, n_images)
        
        # Создаем фигуру
        fig = plt.figure(figsize=(ncols * 4, nrows * 4))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        
        # Отображаем ЧБ изображение
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(grayscale, cmap='gray')
        ax.set_title("Исходное ЧБ")
        ax.axis('off')
        
        # Отображаем варианты колоризации
        for i, (variant, title) in enumerate(zip(variants, titles)):
            ax = fig.add_subplot(gs[0, i+1])
            ax.imshow(variant)
            ax.set_title(title)
            ax.axis('off')
        
        # Отображаем карты неопределенности, если предоставлены
        if uncertainty is not None:
            for i, uncert in enumerate(uncertainty):
                ax = fig.add_subplot(gs[1, i+1])
                cmap = plt.cm.viridis
                im = ax.imshow(uncert, cmap=cmap)
                ax.set_title(f"Неопределенность {i+1}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "comparisons", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Варианты колоризации сохранены: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def create_progression_visualization(self,
                                        grayscale: np.ndarray,
                                        steps: List[np.ndarray],
                                        titles: Optional[List[str]] = None,
                                        filename: Optional[str] = None) -> Image.Image:
        """
        Создает визуализацию прогрессии колоризации.
        
        Args:
            grayscale (np.ndarray): ЧБ изображение
            steps (List[np.ndarray]): Список шагов колоризации
            titles (List[str], optional): Список заголовков для шагов
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Визуализация прогрессии колоризации
        """
        # Определяем количество шагов
        n_steps = len(steps)
        
        # Генерируем заголовки, если не предоставлены
        if titles is None:
            titles = [f"Шаг {i+1}" for i in range(n_steps)]
        
        # Определяем количество строк и столбцов
        nrows = 1
        ncols = n_steps + 1  # +1 для ЧБ изображения
        
        # Создаем фигуру
        fig = plt.figure(figsize=(ncols * 4, nrows * 4))
        
        # Отображаем ЧБ изображение
        ax = fig.add_subplot(1, ncols, 1)
        ax.imshow(grayscale, cmap='gray')
        ax.set_title("Исходное ЧБ")
        ax.axis('off')
        
        # Отображаем шаги колоризации
        for i, (step, title) in enumerate(zip(steps, titles)):
            ax = fig.add_subplot(1, ncols, i+2)
            ax.imshow(step)
            ax.set_title(title)
            ax.axis('off')
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "comparisons", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Прогрессия колоризации сохранена: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def create_style_transfer_comparison(self,
                                       grayscale: np.ndarray,
                                       colorized: np.ndarray,
                                       style_reference: np.ndarray,
                                       stylized: np.ndarray,
                                       filename: Optional[str] = None) -> Image.Image:
        """
        Создает визуализацию переноса стиля.
        
        Args:
            grayscale (np.ndarray): ЧБ изображение
            colorized (np.ndarray): Колоризованное изображение
            style_reference (np.ndarray): Референсное изображение стиля
            stylized (np.ndarray): Стилизованное изображение
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Визуализация переноса стиля
        """
        # Создаем фигуру 2x2
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Отображаем изображения
        axes[0, 0].imshow(grayscale, cmap='gray')
        axes[0, 0].set_title("Исходное ЧБ")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(colorized)
        axes[0, 1].set_title("Базовая колоризация")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(style_reference)
        axes[1, 0].set_title("Референс стиля")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(stylized)
        axes[1, 1].set_title("Стилизованная колоризация")
        axes[1, 1].axis('off')
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "comparisons", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Сравнение стилей сохранено: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img


class GridVisualizer:
    """
    Класс для создания сетки изображений.
    
    Args:
        nrows (int): Количество строк
        ncols (int): Количество столбцов
        figsize (tuple): Размер фигуры (ширина, высота)
        dpi (int): Разрешение для сохранения изображений
    """
    def __init__(self, nrows: int = 1, ncols: int = 2, figsize: Tuple[int, int] = (10, 5), dpi: int = 150):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.dpi = dpi
        
        # Создаем фигуру и сетку
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Преобразуем self.axes в 2D массив, если нужно
        if nrows == 1 and ncols == 1:
            self.axes = np.array([[self.axes]])
        elif nrows == 1 or ncols == 1:
            self.axes = np.array([self.axes]).reshape(nrows, ncols)
    
    def add_image(self, row: int, col: int, image: np.ndarray, title: str = "", cmap: Optional[str] = None):
        """
        Добавляет изображение в сетку.
        
        Args:
            row (int): Индекс строки
            col (int): Индекс столбца
            image (np.ndarray): Изображение
            title (str): Заголовок изображения
            cmap (str, optional): Цветовая карта
        """
        # Проверяем индексы
        if row >= self.nrows or col >= self.ncols:
            raise ValueError(f"Индексы ({row}, {col}) выходят за пределы сетки {self.nrows}x{self.ncols}")
        
        # Отображаем изображение
        self.axes[row, col].imshow(image, cmap=cmap)
        self.axes[row, col].set_title(title)
        self.axes[row, col].axis('off')
    
    def add_colorbar(self, row: int, col: int, mappable):
        """
        Добавляет цветовую шкалу.
        
        Args:
            row (int): Индекс строки
            col (int): Индекс столбца
            mappable: Объект, к которому привязана цветовая шкала
        """
        plt.colorbar(mappable, ax=self.axes[row, col], fraction=0.046, pad=0.04)
    
    def save(self, filename: str, output_dir: str = "./output/comparisons"):
        """
        Сохраняет сетку.
        
        Args:
            filename (str): Имя файла
            output_dir (str): Директория для сохранения
        """
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Определяем путь для сохранения
        save_path = os.path.join(output_dir, filename)
        
        # Создаем директорию, если не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем изображение
        self.fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Сетка сохранена: {save_path}")
    
    def to_image(self) -> Image.Image:
        """
        Преобразует сетку в PIL.Image.
        
        Returns:
            PIL.Image.Image: Изображение с сеткой
        """
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(self.fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def close(self):
        """Закрывает фигуру."""
        plt.close(self.fig)


class BatchVisualizer:
    """
    Класс для визуализации результатов пакетной обработки.
    
    Args:
        output_dir (str): Директория для сохранения результатов
        max_images (int): Максимальное количество изображений в сетке
        dpi (int): Разрешение для сохранения изображений
    """
    def __init__(self, output_dir: str = "./output", max_images: int = 16, dpi: int = 150):
        self.output_dir = output_dir
        self.max_images = max_images
        self.dpi = dpi
        
        # Создаем директории для результатов
        os.makedirs(os.path.join(output_dir, "batch_results"), exist_ok=True)
    
    def create_batch_grid(self,
                         grayscale_batch: torch.Tensor,
                         colorized_batch: torch.Tensor,
                         nrow: int = 4,
                         padding: int = 2,
                         normalize: bool = True,
                         filename: Optional[str] = None) -> Image.Image:
        """
        Создает сетку из пакета изображений.
        
        Args:
            grayscale_batch (torch.Tensor): Пакет ЧБ изображений [B, 1, H, W]
            colorized_batch (torch.Tensor): Пакет колоризованных изображений [B, 3, H, W]
            nrow (int): Количество изображений в строке
            padding (int): Отступ между изображениями
            normalize (bool): Нормализовать ли значения в диапазон [0, 1]
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Сетка изображений
        """
        # Определяем количество изображений
        batch_size = min(grayscale_batch.shape[0], self.max_images)
        
        # Ограничиваем количество изображений
        grayscale_batch = grayscale_batch[:batch_size]
        colorized_batch = colorized_batch[:batch_size]
        
        # Дублируем ЧБ изображения для визуализации
        grayscale_rgb = grayscale_batch.repeat(1, 3, 1, 1)
        
        # Объединяем пакеты
        combined_batch = torch.cat([grayscale_rgb, colorized_batch], dim=0)
        
        # Создаем сетку
        grid = make_grid(combined_batch, nrow=nrow, padding=padding, normalize=normalize)
        
        # Преобразуем в numpy
        grid_np = grid.cpu().numpy()
        grid_np = np.transpose(grid_np, (1, 2, 0))
        
        # Если нужно денормализовать
        if normalize:
            grid_np = (grid_np * 255).astype(np.uint8)
        
        # Создаем PIL.Image
        grid_img = Image.fromarray(grid_np)
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "batch_results", filename)
            grid_img.save(save_path)
            print(f"Сетка сохранена: {save_path}")
        
        return grid_img
    
    def create_batch_comparison(self,
                              grayscale_batch: torch.Tensor,
                              colorized_batch: torch.Tensor,
                              original_batch: Optional[torch.Tensor] = None,
                              indices: Optional[List[int]] = None,
                              max_images: Optional[int] = None,
                              filename: Optional[str] = None) -> Image.Image:
        """
        Создает сравнение из пакета изображений.
        
        Args:
            grayscale_batch (torch.Tensor): Пакет ЧБ изображений [B, 1, H, W]
            colorized_batch (torch.Tensor): Пакет колоризованных изображений [B, 3, H, W]
            original_batch (torch.Tensor, optional): Пакет оригинальных изображений [B, 3, H, W]
            indices (List[int], optional): Индексы изображений для отображения
            max_images (int, optional): Максимальное количество изображений
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Сравнение изображений
        """
        # Определяем максимальное количество изображений
        if max_images is None:
            max_images = min(grayscale_batch.shape[0], self.max_images)
        else:
            max_images = min(max_images, grayscale_batch.shape[0], self.max_images)
        
        # Определяем индексы изображений
        if indices is None:
            indices = list(range(max_images))
        else:
            indices = indices[:max_images]
        
        # Определяем количество изображений и количество столбцов
        n_cols = 2 if original_batch is None else 3
        n_rows = len(indices)
        
        # Создаем фигуру
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        
        # Если только одно изображение, преобразуем оси
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Отображаем изображения
        for i, idx in enumerate(indices):
            # Преобразуем тензоры в numpy
            grayscale = grayscale_batch[idx].cpu().numpy().transpose(1, 2, 0)
            colorized = colorized_batch[idx].cpu().numpy().transpose(1, 2, 0)
            
            # Отображаем ЧБ изображение
            axes[i, 0].imshow(grayscale, cmap='gray')
            axes[i, 0].set_title(f"ЧБ {idx}")
            axes[i, 0].axis('off')
            
            # Отображаем колоризованное изображение
            axes[i, 1].imshow(colorized)
            axes[i, 1].set_title(f"Колоризованное {idx}")
            axes[i, 1].axis('off')
            
            # Отображаем оригинальное изображение, если предоставлено
            if original_batch is not None:
                original = original_batch[idx].cpu().numpy().transpose(1, 2, 0)
                axes[i, 2].imshow(original)
                axes[i, 2].set_title(f"Оригинал {idx}")
                axes[i, 2].axis('off')
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "batch_results", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Сравнение сохранено: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img


class FeatureVisualizer:
    """
    Класс для визуализации внутренних признаков модели.
    
    Args:
        output_dir (str): Директория для сохранения результатов
        dpi (int): Разрешение для сохранения изображений
    """
    def __init__(self, output_dir: str = "./output", dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Создаем директории для результатов
        os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    
    def visualize_feature_maps(self,
                              features: torch.Tensor,
                              nrow: int = 8,
                              cmap: str = 'viridis',
                              max_features: int = 64,
                              filename: Optional[str] = None) -> Image.Image:
        """
        Визуализирует карты признаков.
        
        Args:
            features (torch.Tensor): Карты признаков [B, C, H, W]
            nrow (int): Количество карт в строке
            cmap (str): Цветовая карта
            max_features (int): Максимальное количество карт для отображения
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Визуализация карт признаков
        """
        # Проверяем размерность тензора
        if len(features.shape) != 4:
            raise ValueError("Ожидается 4D тензор [B, C, H, W]")
        
        # Извлекаем карты признаков из первого элемента пакета
        feature_maps = features[0]  # [C, H, W]
        
        # Ограничиваем количество карт
        num_features = min(feature_maps.shape[0], max_features)
        feature_maps = feature_maps[:num_features]
        
        # Нормализуем карты признаков для визуализации
        feature_maps = feature_maps.detach().cpu()
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min() + 1e-8)
        
        # Создаем сетку
        grid = make_grid(feature_maps.unsqueeze(1), nrow=nrow, padding=2, normalize=False)
        
        # Преобразуем в numpy
        grid_np = grid.numpy()[0]  # Берем первый канал, так как карты признаков одноканальные
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Отображаем сетку
        im = ax.imshow(grid_np, cmap=cmap)
        ax.set_title(f"Карты признаков (показано {num_features} из {feature_maps.shape[0]})")
        ax.axis('off')
        
        # Добавляем цветовую шкалу
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "features", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Карты признаков сохранены: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img
    
    def visualize_attention_heads(self,
                                attention_weights: torch.Tensor,
                                nrow: Optional[int] = None,
                                filename: Optional[str] = None) -> Image.Image:
        """
        Визуализирует веса внимания для разных голов.
        
        Args:
            attention_weights (torch.Tensor): Веса внимания [B, H, N, N]
                где H - количество голов, N - количество токенов
            nrow (int, optional): Количество голов в строке
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            PIL.Image.Image: Визуализация весов внимания
        """
        # Проверяем размерность тензора
        if len(attention_weights.shape) != 4:
            raise ValueError("Ожидается 4D тензор [B, H, N, N]")
        
        # Извлекаем веса внимания из первого элемента пакета
        attention = attention_weights[0]  # [H, N, N]
        
        # Определяем количество голов и количество строк
        num_heads = attention.shape[0]
        if nrow is None:
            nrow = int(np.ceil(np.sqrt(num_heads)))
        nrows = int(np.ceil(num_heads / nrow))
        
        # Создаем фигуру
        fig, axes = plt.subplots(nrows, nrow, figsize=(nrow * 3, nrows * 3))
        
        # Если только одна голова, преобразуем оси
        if num_heads == 1:
            axes = np.array([axes])
        
        # Приводим оси к 2D массиву
        if nrows == 1:
            axes = axes.reshape(1, -1)
        elif nrow == 1:
            axes = axes.reshape(-1, 1)
        
        # Отображаем веса внимания для каждой головы
        for h in range(num_heads):
            row = h // nrow
            col = h % nrow
            
            # Получаем веса внимания для текущей головы
            head_attention = attention[h].detach().cpu().numpy()
            
            # Отображаем веса внимания
            im = axes[row, col].imshow(head_attention, cmap='viridis')
            axes[row, col].set_title(f"Голова {h+1}")
            
            # Добавляем цветовую шкалу
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Скрываем оси для пустых ячеек
        for h in range(num_heads, nrows * nrow):
            row = h // nrow
            col = h % nrow
            axes[row, col].axis('off')
        
        # Настраиваем общие параметры
        plt.tight_layout()
        
        # Сохраняем изображение, если указано имя файла
        if filename:
            # Определяем путь для сохранения
            save_path = os.path.join(self.output_dir, "features", filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Веса внимания сохранены: {save_path}")
        
        # Преобразуем в PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        
        return img


# Функции для преобразования и нормализации изображений

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Преобразует тензор в numpy массив.
    
    Args:
        tensor (torch.Tensor): Входной тензор [C, H, W]
        
    Returns:
        np.ndarray: numpy массив [H, W, C]
    """
    # Преобразуем к CPU, если нужно
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Преобразуем в numpy
    array = tensor.detach().numpy()
    
    # Если тензор имеет форму [C, H, W], меняем порядок осей
    if array.ndim == 3 and array.shape[0] in [1, 3, 4]:
        array = np.transpose(array, (1, 2, 0))
    
    # Если это одноканальное изображение, убираем ось канала
    if array.shape[-1] == 1:
        array = array[..., 0]
    
    return array

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Нормализует изображение в диапазон [0, 1] или [0, 255].
    
    Args:
        image (np.ndarray): Входное изображение
        
    Returns:
        np.ndarray: Нормализованное изображение
    """
    # Если значения уже в диапазоне [0, 1] или [0, 255], не меняем
    if image.max() <= 1.0 and image.min() >= 0.0:
        return image
    elif image.max() <= 255.0 and image.min() >= 0.0:
        return image
    
    # Нормализуем в диапазон [0, 1]
    normalized = (image - image.min()) / (image.max() - image.min())
    
    return normalized

def lab_to_rgb(l_channel: torch.Tensor, ab_channels: torch.Tensor) -> np.ndarray:
    """
    Преобразует изображение из пространства Lab в RGB.
    
    Args:
        l_channel (torch.Tensor): Канал L [B, 1, H, W]
        ab_channels (torch.Tensor): Каналы a и b [B, 2, H, W]
        
    Returns:
        np.ndarray: RGB изображение [B, H, W, 3]
    """
    # Проверяем размерности тензоров
    if l_channel.dim() != 4 or ab_channels.dim() != 4:
        raise ValueError("Ожидаются 4D тензоры [B, C, H, W]")
    
    if l_channel.shape[1] != 1 or ab_channels.shape[1] != 2:
        raise ValueError("Неверное количество каналов")
    
    # Объединяем каналы
    lab = torch.cat([l_channel, ab_channels], dim=1)
    
    # Преобразуем в numpy
    batch_size = lab.shape[0]
    result = []
    
    for i in range(batch_size):
        # Получаем изображение Lab
        lab_img = lab[i].detach().cpu().numpy().transpose(1, 2, 0)
        
        # Денормализуем значения
        # L: [0, 1] -> [0, 100]
        # a, b: [-1, 1] -> [-128, 127]
        lab_img[:, :, 0] = lab_img[:, :, 0] * 100.0
        lab_img[:, :, 1:] = lab_img[:, :, 1:] * 127.0
        
        # Преобразуем значения к uint8
        lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)
        
        # Конвертируем Lab в BGR
        bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
        
        # Конвертируем BGR в RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        result.append(rgb_img)
    
    return np.array(result)

def save_image(image: np.ndarray, filename: str, output_dir: str = "./output/colorized"):
    """
    Сохраняет изображение в файл.
    
    Args:
        image (np.ndarray): Изображение
        filename (str): Имя файла
        output_dir (str): Директория для сохранения
    """
    # Создаем директорию, если не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем путь для сохранения
    save_path = os.path.join(output_dir, filename)
    
    # Нормализуем изображение, если нужно
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Сохраняем изображение
    if image.ndim == 2 or image.shape[-1] == 1:
        # Для ЧБ изображений
        pil_img = Image.fromarray(image.squeeze())
        pil_img.save(save_path)
    else:
        # Для цветных изображений
        pil_img = Image.fromarray(image)
        pil_img.save(save_path)
        
    print(f"Изображение сохранено: {save_path}")


# Создаем функцию для инициализации визуализатора
def create_visualizer(output_dir: str = "./output", create_dirs: bool = True, dpi: int = 150) -> ColorizationVisualizer:
    """
    Создает визуализатор результатов колоризации.
    
    Args:
        output_dir (str): Директория для сохранения результатов
        create_dirs (bool): Создавать ли поддиректории для результатов
        dpi (int): Разрешение для сохранения изображений
        
    Returns:
        ColorizationVisualizer: Визуализатор результатов колоризации
    """
    return ColorizationVisualizer(output_dir, create_dirs, dpi)


if __name__ == "__main__":
    # Пример использования модуля визуализации
    
    # Создаем тестовые данные
    grayscale = np.random.rand(256, 256)  # Случайное ЧБ изображение
    colorized = np.random.rand(256, 256, 3)  # Случайное цветное изображение
    original = np.random.rand(256, 256, 3)  # Случайное оригинальное изображение
    uncertainty = np.random.rand(256, 256)  # Случайная карта неопределенности
    
    try:
        # Создаем визуализатор
        visualizer = create_visualizer(output_dir="./output", create_dirs=True)
        
        # Создаем сравнение
        comparison = visualizer.create_comparison(
            grayscale=grayscale,
            colorized=colorized,
            original=original,
            uncertainty=uncertainty,
            filename="test_comparison.png"
        )
        
        # Создаем карту неопределенности
        uncertainty_map = visualizer.create_uncertainty_map(
            uncertainty=uncertainty,
            colorized=colorized,
            filename="test_uncertainty.png"
        )
        
        # Создаем цветовую палитру
        palette, colors = visualizer.create_color_palette(
            image=colorized,
            n_colors=8,
            filename="test_palette.png"
        )
        
        print("Визуализации созданы успешно!")
        
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {e}")