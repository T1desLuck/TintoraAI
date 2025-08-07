"""
Memory Bank: Самообучающаяся база знаний цветов для колоризации.

Данный модуль реализует механизм хранения и использования цветовых решений для
различных объектов, текстур и сцен. Memory Bank действует как самообучающаяся 
база знаний, которая накапливает успешные случаи колоризации и использует их для
улучшения будущих предсказаний, особенно в сложных или неоднозначных случаях.

Ключевые особенности:
- Хранение ассоциаций между семантическими признаками и цветовыми решениями
- Система голосования для выбора наиболее подходящих цветовых кандидатов
- Обновление базы знаний на основе обратной связи о качестве колоризации
- Структура хранения с индексацией для быстрого поиска и извлечения
- Компрессия и дистилляция знаний для эффективного хранения

Преимущества для колоризации:
- Повышение консистентности цветов для похожих объектов
- Снижение неопределенности в выборе цветов для неоднозначных случаев
- Возможность использовать предыдущий опыт для улучшения новых колоризаций
- Адаптация к предпочтениям пользователя и конкретным доменам
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, OrderedDict, deque
import os
import time
import json
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime


class FaissIndexWrapper:
    """
    Обертка для индекса FAISS для векторного поиска ближайших соседей.
    
    Args:
        dim (int): Размерность векторов
        index_type (str): Тип индекса FAISS ('flat', 'ivf', 'hnsw')
        gpu_id (int): ID GPU для использования или -1 для CPU
        max_items (int): Максимальное количество элементов в индексе
    """
    def __init__(self, dim=512, index_type='flat', gpu_id=-1, max_items=1000000):
        self.dim = dim
        self.index_type = index_type
        self.gpu_id = gpu_id
        self.max_items = max_items
        self.is_gpu = False
        self.needs_training = False
        
        # Попытка импортировать FAISS
        try:
            import faiss
            self.faiss = faiss
            self.has_faiss = True
            self.create_index()
        except ImportError:
            print("FAISS не установлен. Используем альтернативную реализацию на основе PyTorch.")
            self.has_faiss = False
            self.create_pytorch_index()
        
    def create_index(self):
        """
        Создает индекс FAISS в зависимости от указанного типа.
        """
        if not self.has_faiss:
            return self.create_pytorch_index()
            
        if self.index_type == 'flat':
            # Простой плоский индекс, точный поиск
            self.index = self.faiss.IndexFlatL2(self.dim)
        elif self.index_type == 'ivf':
            # IVF индекс для более быстрого поиска
            quantizer = self.faiss.IndexFlatL2(self.dim)
            nlist = min(4096, self.max_items // 10)  # Количество центроидов
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dim, nlist, self.faiss.METRIC_L2)
            self.index.nprobe = 256  # Количество ячеек для проверки при поиске
            self.needs_training = True
        elif self.index_type == 'hnsw':
            # HNSW индекс для еще более быстрого поиска
            self.index = self.faiss.IndexHNSWFlat(self.dim, 32)  # 32 соседа на уровень
            self.index.hnsw.efConstruction = 40  # Больше - точнее, но медленнее
            self.index.hnsw.efSearch = 64
        else:
            raise ValueError(f"Неподдерживаемый тип индекса: {self.index_type}")
        
        # Перемещаем индекс на GPU, если указан
        if self.gpu_id >= 0 and self.faiss.get_num_gpus() > 0:
            try:
                res = self.faiss.StandardGpuResources()
                self.index = self.faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
                self.is_gpu = True
            except Exception as e:
                print(f"Не удалось переместить индекс на GPU: {e}")
                self.is_gpu = False
                
    def create_pytorch_index(self):
        """
        Создает простую альтернативу FAISS на основе PyTorch.
        """
        self.vectors = []
        self.ids = []
        
    def train(self, vectors):
        """
        Обучает индекс на основе набора векторов.
        
        Args:
            vectors (np.ndarray): Массив векторов [N, dim]
        """
        if not self.has_faiss:
            return
            
        if self.needs_training and vectors.shape[0] > 0:
            self.index.train(vectors)
            self.needs_training = False
            
    def add(self, vectors, ids=None):
        """
        Добавляет векторы в индекс.
        
        Args:
            vectors (np.ndarray): Массив векторов [N, dim]
            ids (np.ndarray, optional): ID для векторов
        """
        if vectors.shape[0] == 0:
            return
            
        if not self.has_faiss:
            # Реализация на PyTorch
            vectors_tensor = torch.from_numpy(vectors) if isinstance(vectors, np.ndarray) else vectors
            
            if ids is None:
                ids = np.arange(len(self.vectors), len(self.vectors) + vectors.shape[0])
                
            for i in range(vectors.shape[0]):
                self.vectors.append(vectors_tensor[i].view(1, -1))
                self.ids.append(ids[i])
            return
            
        # FAISS реализация
        # Обучаем индекс, если нужно
        if self.needs_training and vectors.shape[0] >= 1000:
            self.train(vectors)
            
        # Добавляем векторы
        if ids is not None:
            self.index.add_with_ids(vectors, ids)
        else:
            self.index.add(vectors)
            
    def search(self, query_vectors, k=5):
        """
        Ищет ближайших соседей для запроса.
        
        Args:
            query_vectors (np.ndarray): Запрос [N, dim]
            k (int): Количество ближайших соседей
            
        Returns:
            tuple: (расстояния [N, k], индексы [N, k])
        """
        if not self.has_faiss:
            # Реализация на PyTorch
            query_tensor = torch.from_numpy(query_vectors) if isinstance(query_vectors, np.ndarray) else query_vectors
            n_queries = query_tensor.shape[0]
            
            # Подготавливаем массивы для результатов
            distances = torch.zeros((n_queries, min(k, len(self.vectors))))
            indices = torch.zeros((n_queries, min(k, len(self.vectors))), dtype=torch.int64)
            
            if len(self.vectors) == 0:
                return distances.numpy(), indices.numpy()
                
            for i in range(n_queries):
                if len(self.vectors) == 0:
                    continue
                    
                # Объединяем все векторы в один тензор
                vectors_tensor = torch.cat(self.vectors, dim=0)
                
                # Вычисляем L2 расстояния
                dists = torch.cdist(query_tensor[i].view(1, -1), vectors_tensor)[0]
                
                # Находим ближайшие
                k_actual = min(k, len(self.vectors))
                values, indices_local = torch.topk(dists, k_actual, largest=False)
                
                # Сохраняем результаты
                distances[i, :k_actual] = values
                for j, idx in enumerate(indices_local):
                    indices[i, j] = self.ids[idx]
                    
            return distances.numpy(), indices.numpy()
            
        # FAISS реализация
        return self.index.search(query_vectors, k)
    
    def get_size(self):
        """
        Возвращает количество элементов в индексе.
        
        Returns:
            int: Количество элементов
        """
        if not self.has_faiss:
            return len(self.vectors)
            
        return self.index.ntotal
    
    def save(self, path):
        """
        Сохраняет индекс на диск.
        
        Args:
            path (str): Путь для сохранения
        """
        if not self.has_faiss:
            # Реализация на PyTorch
            vectors_data = [v.numpy() if isinstance(v, torch.Tensor) else v for v in self.vectors]
            data = {
                'vectors': vectors_data,
                'ids': self.ids
            }
            torch.save(data, path)
            return
            
        # FAISS реализация
        # Если индекс на GPU, переносим его на CPU
        if self.is_gpu:
            index_cpu = self.faiss.index_gpu_to_cpu(self.index)
            self.faiss.write_index(index_cpu, path)
        else:
            self.faiss.write_index(self.index, path)
            
    def load(self, path):
        """
        Загружает индекс с диска.
        
        Args:
            path (str): Путь для загрузки
        """
        if not self.has_faiss:
            # Реализация на PyTorch
            data = torch.load(path)
            self.vectors = [torch.tensor(v) if isinstance(v, np.ndarray) else v for v in data['vectors']]
            self.ids = data['ids']
            return
            
        # FAISS реализация
        # Загружаем индекс с диска
        self.index = self.faiss.read_index(path)
        
        # Перемещаем на GPU, если нужно
        if self.gpu_id >= 0 and self.faiss.get_num_gpus() > 0:
            try:
                res = self.faiss.StandardGpuResources()
                self.index = self.faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
                self.is_gpu = True
            except Exception as e:
                print(f"Не удалось переместить индекс на GPU: {e}")
                self.is_gpu = False
                
        self.needs_training = False


class ColorEncoder(nn.Module):
    """
    Энкодер для извлечения признаков из цветных изображений.
    
    Args:
        input_channels (int): Количество входных каналов
        output_dim (int): Размерность выходного вектора
    """
    def __init__(self, input_channels=3, output_dim=512):
        super(ColorEncoder, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        """
        Извлекает признаки из изображения.
        
        Args:
            x (torch.Tensor): Входное изображение [B, C, H, W]
            
        Returns:
            torch.Tensor: Признаки [B, output_dim]
        """
        return self.encoder(x)


class GrayToColorPredictor(nn.Module):
    """
    Предсказатель цвета на основе черно-белого изображения.
    
    Args:
        input_channels (int): Количество входных каналов (обычно 1 для ЧБ)
        output_channels (int): Количество выходных каналов (обычно 2 для ab)
        feature_dim (int): Размерность признаков
    """
    def __init__(self, input_channels=1, output_channels=2, feature_dim=512):
        super(GrayToColorPredictor, self).__init__()
        
        # Энкодер для извлечения признаков из ЧБ изображения
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Извлечение признаков на уровне всего изображения
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, feature_dim)
        )
        
        # Декодер для генерации карты цветов
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Генерирует цвета на основе ЧБ изображения.
        
        Args:
            x (torch.Tensor): ЧБ изображение [B, 1, H, W]
            
        Returns:
            dict: {
                'color_map': torch.Tensor,  # Карта цветов [B, output_channels, H, W]
                'features': torch.Tensor  # Извлеченные признаки [B, feature_dim]
            }
        """
        # Извлекаем признаки
        features_map = self.encoder(x)
        
        # Получаем глобальные признаки
        global_features = self.global_pool(features_map)
        
        # Генерируем карту цветов
        color_map = self.decoder(features_map)
        
        return {
            'color_map': color_map,
            'features': global_features
        }


class MemoryItem:
    """
    Элемент для хранения в банке памяти.
    
    Args:
        features (np.ndarray): Признаки [feature_dim]
        color (np.ndarray): Цвет [color_channels, H, W]
        metadata (dict): Метаданные (опционально)
        quality (float): Оценка качества (0.0 - 1.0)
    """
    def __init__(self, features, color, metadata=None, quality=0.5):
        self.features = features
        self.color = color
        self.metadata = metadata if metadata is not None else {}
        self.quality = quality
        self.usage_count = 0
        self.last_used = time.time()
        self.creation_time = time.time()
        
    def update_usage(self):
        """Обновляет статистику использования элемента."""
        self.usage_count += 1
        self.last_used = time.time()
        
    def update_quality(self, new_quality):
        """
        Обновляет оценку качества элемента.
        
        Args:
            new_quality (float): Новая оценка качества
        """
        # Скользящее среднее для оценки качества
        self.quality = 0.8 * self.quality + 0.2 * new_quality
        
    def to_dict(self):
        """
        Преобразует элемент в словарь для сериализации.
        
        Returns:
            dict: Словарь с данными элемента
        """
        return {
            'features': self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            'color': self.color.tolist() if isinstance(self.color, np.ndarray) else self.color,
            'metadata': self.metadata,
            'quality': self.quality,
            'usage_count': self.usage_count,
            'last_used': self.last_used,
            'creation_time': self.creation_time
        }
        
    @classmethod
    def from_dict(cls, data):
        """
        Создает элемент из словаря.
        
        Args:
            data (dict): Словарь с данными элемента
            
        Returns:
            MemoryItem: Элемент банка памяти
        """
        features = np.array(data['features']) if not isinstance(data['features'], np.ndarray) else data['features']
        color = np.array(data['color']) if not isinstance(data['color'], np.ndarray) else data['color']
        
        item = cls(features, color, data['metadata'], data['quality'])
        item.usage_count = data['usage_count']
        item.last_used = data['last_used']
        item.creation_time = data['creation_time']
        return item


class MemoryBank(nn.Module):
    """
    Банк памяти для хранения и использования цветовых решений.
    
    Args:
        feature_dim (int): Размерность признаков
        color_channels (int): Количество цветовых каналов
        max_items (int): Максимальное количество элементов
        index_type (str): Тип индекса для поиска ('flat', 'ivf', 'hnsw')
        gpu_id (int): ID GPU для использования или -1 для CPU
        save_dir (str): Директория для сохранения банка памяти
    """
    def __init__(self, feature_dim=512, color_channels=2, max_items=100000, 
                 index_type='flat', gpu_id=-1, save_dir='./data/memory_bank'):
        super(MemoryBank, self).__init__()
        
        self.feature_dim = feature_dim
        self.color_channels = color_channels
        self.max_items = max_items
        self.index_type = index_type
        self.gpu_id = gpu_id
        self.save_dir = save_dir
        
        # Создаем индекс для быстрого поиска
        self.index = FaissIndexWrapper(
            dim=feature_dim,
            index_type=index_type,
            gpu_id=gpu_id,
            max_items=max_items
        )
        
        # Словарь для хранения элементов
        self.items = OrderedDict()
        
        # Счетчик для генерации ID
        self.next_id = 0
        
        # Для отслеживания статистик
        self.register_buffer('total_queries', torch.zeros(1, dtype=torch.long))
        self.register_buffer('successful_queries', torch.zeros(1, dtype=torch.long))
        self.register_buffer('total_items_added', torch.zeros(1, dtype=torch.long))
        self.register_buffer('total_items_removed', torch.zeros(1, dtype=torch.long))
        
        # Создаем директорию для сохранения, если её нет
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
    @property
    def device(self):
        """Возвращает устройство, на котором находится модуль."""
        return next(self.parameters()).device
            
    def add_item(self, features, color, metadata=None, quality=0.5):
        """
        Добавляет элемент в банк памяти.
        
        Args:
            features (torch.Tensor): Признаки [B, feature_dim]
            color (torch.Tensor): Цвет [B, color_channels, H, W]
            metadata (dict, optional): Метаданные
            quality (float): Оценка качества (0.0 - 1.0)
            
        Returns:
            list: Список ID добавленных элементов
        """
        batch_size = features.shape[0]
        added_ids = []
        
        # Преобразуем в numpy для работы с FAISS
        features_np = features.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        
        # Добавляем каждый элемент пакета
        for i in range(batch_size):
            # Проверяем, не слишком ли много элементов
            if len(self.items) >= self.max_items:
                # Удаляем наименее полезные элементы
                self._remove_least_useful_items(batch_size)
                
            # Генерируем ID
            item_id = self.next_id
            self.next_id += 1
            
            # Создаем метаданные, если не предоставлены
            if metadata is None:
                item_metadata = {'timestamp': datetime.now().isoformat()}
            else:
                item_metadata = metadata.copy()
                item_metadata['timestamp'] = datetime.now().isoformat()
            
            # Создаем элемент
            item = MemoryItem(
                features=features_np[i],
                color=color_np[i],
                metadata=item_metadata,
                quality=quality
            )
            
            # Добавляем в словарь
            self.items[item_id] = item
            
            # Добавляем в индекс
            self.index.add(features_np[i].reshape(1, -1), np.array([item_id]))
            
            added_ids.append(item_id)
            
        # Обновляем статистику
        self.total_items_added += batch_size
        
        return added_ids
    
    def _remove_least_useful_items(self, count=1):
        """
        Удаляет наименее полезные элементы из банка памяти.
        
        Args:
            count (int): Количество элементов для удаления
        """
        # Если элементов меньше, чем нужно удалить, просто выходим
        if len(self.items) <= count:
            return
            
        # Вычисляем "полезность" каждого элемента
        # Полезность = quality * (usage_count + 1) / (time_since_last_used + 1)
        current_time = time.time()
        item_utilities = {}
        
        for item_id, item in self.items.items():
            time_since_last_used = current_time - item.last_used
            utility = item.quality * (item.usage_count + 1) / (time_since_last_used + 1)
            item_utilities[item_id] = utility
            
        # Сортируем элементы по полезности
        sorted_items = sorted(item_utilities.items(), key=lambda x: x[1])
        
        # Удаляем наименее полезные элементы
        items_to_remove = sorted_items[:count]
        for item_id, _ in items_to_remove:
            del self.items[item_id]
            
        # Обновляем статистику
        self.total_items_removed += count
        
        # Примечание: мы не удаляем элементы из индекса FAISS, так как это
        # сложная операция. Вместо этого мы просто игнорируем результаты поиска
        # для ID, которых нет в словаре self.items
        
    def query(self, features, k=5, return_distances=False):
        """
        Ищет ближайшие элементы в банке памяти.
        
        Args:
            features (torch.Tensor): Признаки для поиска [B, feature_dim]
            k (int): Количество ближайших соседей
            return_distances (bool): Возвращать ли расстояния
            
        Returns:
            dict: {
                'colors': torch.Tensor,  # Цвета [B, k, color_channels, H, W]
                'qualities': torch.Tensor,  # Оценки качества [B, k]
                'ids': list,  # ID найденных элементов
                'distances': torch.Tensor (опционально)  # Расстояния [B, k]
            }
        """
        batch_size = features.shape[0]
        self.total_queries += batch_size
        
        # Если банк памяти пуст, возвращаем None
        if len(self.items) == 0:
            return None
            
        # Преобразуем в numpy для работы с FAISS
        features_np = features.detach().cpu().numpy()
        
        # Ищем ближайших соседей
        distances, indices = self.index.search(features_np, min(k, len(self.items)))
        
        # Подготавливаем списки для результатов
        all_colors = []
        all_qualities = []
        all_ids = []
        
        # Обрабатываем результаты
        for i in range(batch_size):
            batch_colors = []
            batch_qualities = []
            batch_ids = []
            
            for j in range(min(k, len(self.items))):
                if j < indices.shape[1]:
                    item_id = int(indices[i, j])
                    
                    # Проверяем, что элемент все еще существует
                    if item_id in self.items:
                        item = self.items[item_id]
                        batch_colors.append(item.color)
                        batch_qualities.append(item.quality)
                        batch_ids.append(item_id)
                        
                        # Обновляем статистику использования
                        item.update_usage()
                    else:
                        # Элемент был удален, добавляем заглушки
                        batch_colors.append(np.zeros((self.color_channels, 1, 1), dtype=np.float32))
                        batch_qualities.append(0.0)
                        batch_ids.append(-1)
            
            # Если нашли элементы, увеличиваем счетчик успешных запросов
            if len(batch_ids) > 0:
                self.successful_queries += 1
                
            all_colors.append(batch_colors)
            all_qualities.append(batch_qualities)
            all_ids.append(batch_ids)
            
        # Преобразуем в тензоры
        colors_tensor = torch.tensor(np.array(all_colors), device=self.device)
        qualities_tensor = torch.tensor(np.array(all_qualities), device=self.device)
        
        # Подготавливаем результат
        result = {
            'colors': colors_tensor,
            'qualities': qualities_tensor,
            'ids': all_ids
        }
        
        if return_distances:
            result['distances'] = torch.tensor(distances, device=self.device)
            
        return result
    
    def update_item_quality(self, item_id, quality):
        """
        Обновляет оценку качества элемента.
        
        Args:
            item_id (int): ID элемента
            quality (float): Новая оценка качества
        """
        if item_id in self.items:
            self.items[item_id].update_quality(quality)
            
    def get_item(self, item_id):
        """
        Получает элемент по ID.
        
        Args:
            item_id (int): ID элемента
            
        Returns:
            MemoryItem: Элемент или None, если не найден
        """
        return self.items.get(item_id, None)
    
    def save(self, filename=None):
        """
        Сохраняет банк памяти на диск.
        
        Args:
            filename (str, optional): Имя файла для сохранения.
                Если None, используется текущая дата и время.
        """
        if filename is None:
            # Генерируем имя файла на основе текущей даты и времени
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"memory_bank_{timestamp}"
            
        # Полный путь для сохранения
        save_path = os.path.join(self.save_dir, filename)
        index_path = f"{save_path}_index"
        items_path = f"{save_path}_items.json"
        
        # Сохраняем индекс
        self.index.save(index_path)
        
        # Сохраняем элементы
        with open(items_path, 'w') as f:
            # Преобразуем элементы в словарь для сериализации
            items_dict = {str(item_id): item.to_dict() for item_id, item in self.items.items()}
            json.dump({
                'items': items_dict,
                'next_id': self.next_id,
                'stats': {
                    'total_queries': self.total_queries.item(),
                    'successful_queries': self.successful_queries.item(),
                    'total_items_added': self.total_items_added.item(),
                    'total_items_removed': self.total_items_removed.item()
                }
            }, f)
            
        print(f"Банк памяти сохранен: {save_path}")
        
    def load(self, filename):
        """
        Загружает банк памяти с диска.
        
        Args:
            filename (str): Имя файла для загрузки (без расширения)
            
        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        # Полный путь для загрузки
        load_path = os.path.join(self.save_dir, filename)
        index_path = f"{load_path}_index"
        items_path = f"{load_path}_items.json"
        
        try:
            # Загружаем индекс
            self.index.load(index_path)
            
            # Загружаем элементы
            with open(items_path, 'r') as f:
                data = json.load(f)
                
                # Восстанавливаем элементы
                self.items = OrderedDict()
                for item_id_str, item_data in data['items'].items():
                    item_id = int(item_id_str)
                    self.items[item_id] = MemoryItem.from_dict(item_data)
                    
                # Восстанавливаем счетчик ID
                self.next_id = data['next_id']
                
                # Восстанавливаем статистику
                stats = data['stats']
                self.total_queries.fill_(stats['total_queries'])
                self.successful_queries.fill_(stats['successful_queries'])
                self.total_items_added.fill_(stats['total_items_added'])
                self.total_items_removed.fill_(stats['total_items_removed'])
                
            print(f"Банк памяти загружен: {load_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке банка памяти: {e}")
            return False
            
    def get_stats(self):
        """
        Возвращает статистики банка памяти.
        
        Returns:
            dict: Статистики банка памяти
        """
        # Вычисляем процент успешных запросов
        success_rate = 0.0
        if self.total_queries.item() > 0:
            success_rate = self.successful_queries.item() / self.total_queries.item()
            
        return {
            'total_items': len(self.items),
            'total_queries': self.total_queries.item(),
            'successful_queries': self.successful_queries.item(),
            'success_rate': success_rate,
            'total_items_added': self.total_items_added.item(),
            'total_items_removed': self.total_items_removed.item()
        }
        
    def get_quality_distribution(self):
        """
        Возвращает распределение оценок качества элементов.
        
        Returns:
            dict: Распределение оценок качества
        """
        if len(self.items) == 0:
            return {'bins': [], 'counts': []}
            
        # Собираем оценки качества
        qualities = [item.quality for item in self.items.values()]
        
        # Создаем гистограмму
        bins = np.linspace(0.0, 1.0, 11)  # 10 бинов от 0.0 до 1.0
        hist, _ = np.histogram(qualities, bins=bins)
        
        return {
            'bins': bins.tolist(),
            'counts': hist.tolist()
        }


class VotingColorSelector(nn.Module):
    """
    Селектор цвета на основе голосования.
    
    Args:
        color_channels (int): Количество цветовых каналов
        use_weighted_voting (bool): Использовать ли взвешенное голосование
    """
    def __init__(self, color_channels=2, use_weighted_voting=True):
        super(VotingColorSelector, self).__init__()
        
        self.color_channels = color_channels
        self.use_weighted_voting = use_weighted_voting
        
    def forward(self, colors, qualities=None, distances=None):
        """
        Выбирает цвет на основе голосования.
        
        Args:
            colors (torch.Tensor): Цвета [B, N, color_channels, H, W]
            qualities (torch.Tensor, optional): Оценки качества [B, N]
            distances (torch.Tensor, optional): Расстояния [B, N]
            
        Returns:
            torch.Tensor: Выбранный цвет [B, color_channels, H, W]
        """
        batch_size, num_candidates = colors.shape[:2]
        
        # Если только один кандидат, возвращаем его
        if num_candidates == 1:
            return colors[:, 0]
            
        # Определяем веса для голосования
        if self.use_weighted_voting and qualities is not None and distances is not None:
            # Веса на основе качества и расстояния
            # Нормализуем расстояния
            max_distance = torch.max(distances, dim=1, keepdim=True)[0] + 1e-8
            normalized_distances = distances / max_distance
            
            # Комбинируем качество и обратное расстояние
            weights = qualities * (1 - normalized_distances)
            
            # Нормализуем веса
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
            
            # Расширяем размерность весов для умножения на цвета
            weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            # Взвешенное среднее цветов
            weighted_colors = colors * weights
            selected_color = torch.sum(weighted_colors, dim=1)
        elif self.use_weighted_voting and qualities is not None:
            # Веса только на основе качества
            # Нормализуем веса
            weights = qualities / (torch.sum(qualities, dim=1, keepdim=True) + 1e-8)
            
            # Расширяем размерность весов для умножения на цвета
            weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            # Взвешенное среднее цветов
            weighted_colors = colors * weights
            selected_color = torch.sum(weighted_colors, dim=1)
        else:
            # Простое среднее цветов
            selected_color = torch.mean(colors, dim=1)
            
        return selected_color


class ColorFusionModule(nn.Module):
    """
    Модуль для слияния цвета из банка памяти с предсказанным цветом.
    
    Args:
        input_channels (int): Количество входных каналов для базовых признаков
        color_channels (int): Количество цветовых каналов
    """
    def __init__(self, input_channels=512, color_channels=2):
        super(ColorFusionModule, self).__init__()
        
        # Сеть для генерации коэффициентов слияния
        self.fusion_network = nn.Sequential(
            nn.Conv2d(input_channels + color_channels * 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, predicted_color, memory_color):
        """
        Сливает предсказанный цвет и цвет из банка памяти.
        
        Args:
            features (torch.Tensor): Базовые признаки [B, C, H, W]
            predicted_color (torch.Tensor): Предсказанный цвет [B, color_channels, H, W]
            memory_color (torch.Tensor): Цвет из банка памяти [B, color_channels, H, W]
            
        Returns:
            dict: {
                'fused_color': torch.Tensor,  # Слитый цвет [B, color_channels, H, W]
                'fusion_weights': torch.Tensor  # Веса слияния [B, 1, H, W]
            }
        """
        # Изменяем размер цветовых карт, если нужно
        if predicted_color.shape[2:] != features.shape[2:]:
            predicted_color = F.interpolate(
                predicted_color, size=features.shape[2:], mode='bilinear', align_corners=False
            )
            
        if memory_color.shape[2:] != features.shape[2:]:
            memory_color = F.interpolate(
                memory_color, size=features.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Объединяем признаки и цвета для генерации весов
        combined_features = torch.cat([features, predicted_color, memory_color], dim=1)
        
        # Генерируем веса слияния
        fusion_weights = self.fusion_network(combined_features)
        
        # Сливаем цвета
        fused_color = fusion_weights * predicted_color + (1 - fusion_weights) * memory_color
        
        return {
            'fused_color': fused_color,
            'fusion_weights': fusion_weights
        }


class MemoryBankModule(nn.Module):
    """
    Модуль, объединяющий все компоненты банка памяти.
    
    Args:
        feature_dim (int): Размерность признаков
        color_channels (int): Количество цветовых каналов
        max_items (int): Максимальное количество элементов в банке памяти
        index_type (str): Тип индекса ('flat', 'ivf', 'hnsw')
        gpu_id (int): ID GPU для использования или -1 для CPU
        save_dir (str): Директория для сохранения банка памяти
        use_fusion (bool): Использовать ли слияние цветов
        k_neighbors (int): Количество ближайших соседей для запроса
    """
    def __init__(self, feature_dim=512, color_channels=2, max_items=100000,
                 index_type='flat', gpu_id=-1, save_dir='./data/memory_bank',
                 use_fusion=True, k_neighbors=5):
        super(MemoryBankModule, self).__init__()
        
        # Банк памяти
        self.memory_bank = MemoryBank(
            feature_dim=feature_dim,
            color_channels=color_channels,
            max_items=max_items,
            index_type=index_type,
            gpu_id=gpu_id,
            save_dir=save_dir
        )
        
        # Энкодер для извлечения признаков из цветных изображений
        self.color_encoder = ColorEncoder(
            input_channels=color_channels + 1,  # L + ab
            output_dim=feature_dim
        )
        
        # Предсказатель цвета на основе ЧБ изображения
        self.color_predictor = GrayToColorPredictor(
            input_channels=1,
            output_channels=color_channels,
            feature_dim=feature_dim
        )
        
        # Селектор цвета на основе голосования
        self.color_selector = VotingColorSelector(
            color_channels=color_channels,
            use_weighted_voting=True
        )
        
        # Модуль слияния цветов
        self.use_fusion = use_fusion
        if use_fusion:
            self.color_fusion = ColorFusionModule(
                input_channels=512,
                color_channels=color_channels
            )
            
        # Количество ближайших соседей для запроса
        self.k_neighbors = k_neighbors
        
        # Для отслеживания эффективности банка памяти
        self.register_buffer('memory_hit_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_miss_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('fusion_quality_sum', torch.zeros(1))
        self.register_buffer('fusion_count', torch.zeros(1, dtype=torch.long))
        
    def encode_image(self, gray_image, color_image=None):
        """
        Извлекает признаки из изображений.
        
        Args:
            gray_image (torch.Tensor): ЧБ изображение [B, 1, H, W]
            color_image (torch.Tensor, optional): Цветное изображение [B, C, H, W]
            
        Returns:
            dict: {
                'features': torch.Tensor,  # Признаки [B, feature_dim]
                'color_map': torch.Tensor (опционально)  # Предсказанная карта цветов
            }
        """
        result = {}
        
        # Если предоставлено цветное изображение, кодируем его вместе с ЧБ
        if color_image is not None:
            # Объединяем ЧБ и цветное изображения
            combined = torch.cat([gray_image, color_image], dim=1)
            
            # Извлекаем признаки
            features = self.color_encoder(combined)
            
            result['features'] = features
        else:
            # Кодируем только ЧБ изображение и предсказываем цвета
            prediction_result = self.color_predictor(gray_image)
            
            result['features'] = prediction_result['features']
            result['color_map'] = prediction_result['color_map']
            
        return result
    
    def add_to_memory(self, gray_image, color_image, metadata=None, quality=0.5):
        """
        Добавляет пару изображений в банк памяти.
        
        Args:
            gray_image (torch.Tensor): ЧБ изображение [B, 1, H, W]
            color_image (torch.Tensor): Цветное изображение [B, C, H, W]
            metadata (dict, optional): Метаданные
            quality (float): Оценка качества (0.0 - 1.0)
            
        Returns:
            list: Список ID добавленных элементов
        """
        # Извлекаем признаки
        encoding_result = self.encode_image(gray_image, color_image)
        
        # Добавляем в банк памяти
        item_ids = self.memory_bank.add_item(
            features=encoding_result['features'],
            color=color_image,
            metadata=metadata,
            quality=quality
        )
        
        return item_ids
    
    def query_memory(self, features, k=None):
        """
        Запрашивает банк памяти на основе признаков.
        
        Args:
            features (torch.Tensor): Признаки для запроса [B, feature_dim]
            k (int, optional): Количество ближайших соседей
            
        Returns:
            dict: Результат запроса или None, если банк памяти пуст
        """
        if k is None:
            k = self.k_neighbors
            
        # Запрашиваем банк памяти
        result = self.memory_bank.query(features, k=k, return_distances=True)
        
        # Обновляем статистику
        if result is not None:
            self.memory_hit_count += features.size(0)
        else:
            self.memory_miss_count += features.size(0)
            
        return result
    
    def fuse_colors(self, features, predicted_color, memory_color):
        """
        Сливает предсказанный цвет и цвет из банка памяти.
        
        Args:
            features (torch.Tensor): Признаки [B, C, H, W]
            predicted_color (torch.Tensor): Предсказанный цвет [B, color_channels, H, W]
            memory_color (torch.Tensor): Цвет из банка памяти [B, color_channels, H, W]
            
        Returns:
            torch.Tensor: Слитый цвет [B, color_channels, H, W]
        """
        if self.use_fusion:
            fusion_result = self.color_fusion(features, predicted_color, memory_color)
            return fusion_result['fused_color']
        else:
            # Простое смешивание с весами 0.5
            return 0.5 * predicted_color + 0.5 * memory_color
    
    def forward(self, gray_image, reference_image=None):
        """
        Выполняет колоризацию с использованием банка памяти.
        
        Args:
            gray_image (torch.Tensor): ЧБ изображение [B, 1, H, W]
            reference_image (torch.Tensor, optional): Референсное изображение [B, C, H, W]
            
        Returns:
            dict: {
                'colorized': torch.Tensor,  # Колоризованное изображение [B, C, H, W]
                'memory_color': torch.Tensor (опционально),  # Цвет из банка памяти
                'predicted_color': torch.Tensor,  # Предсказанный цвет
                'fusion_weights': torch.Tensor (опционально)  # Веса слияния
            }
        """
        # Извлекаем признаки и предсказываем цвет
        result = self.encode_image(gray_image)
        features = result['features']
        predicted_color = result['color_map']
        
        # Запрашиваем банк памяти
        memory_result = self.query_memory(features)
        
        # Если банк памяти пуст или запрос не удался, возвращаем только предсказанный цвет
        if memory_result is None:
            return {
                'colorized': torch.cat([gray_image, predicted_color], dim=1),
                'predicted_color': predicted_color
            }
            
        # Выбираем цвет из банка памяти на основе голосования
        memory_color = self.color_selector(
            memory_result['colors'],
            memory_result['qualities'],
            memory_result['distances']
        )
        
        # Сливаем предсказанный цвет и цвет из банка памяти
        encoder_features = self.color_predictor.encoder(gray_image)
        fused_color = self.fuse_colors(encoder_features, predicted_color, memory_color)
        
        # Возвращаем результат
        result = {
            'colorized': torch.cat([gray_image, fused_color], dim=1),
            'memory_color': memory_color,
            'predicted_color': predicted_color,
            'memory_query_result': memory_result
        }
        
        if self.use_fusion:
            fusion_result = self.color_fusion(encoder_features, predicted_color, memory_color)
            result['fusion_weights'] = fusion_result['fusion_weights']
            
        return result
    
    def update_feedback(self, item_ids, quality_scores):
        """
        Обновляет оценки качества элементов на основе обратной связи.
        
        Args:
            item_ids (list): Список ID элементов
            quality_scores (list): Список оценок качества
        """
        for item_id, quality in zip(item_ids, quality_scores):
            self.memory_bank.update_item_quality(item_id, quality)
            
    def save_memory_bank(self, filename=None):
        """
        Сохраняет банк памяти на диск.
        
        Args:
            filename (str, optional): Имя файла для сохранения
        """
        self.memory_bank.save(filename)
        
    def load_memory_bank(self, filename):
        """
        Загружает банк памяти с диска.
        
        Args:
            filename (str): Имя файла для загрузки
            
        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        return self.memory_bank.load(filename)
    
    def get_stats(self):
        """
        Возвращает статистики модуля.
        
        Returns:
            dict: Статистики модуля
        """
        memory_stats = self.memory_bank.get_stats()
        
        # Вычисляем процент попаданий в банк памяти
        total_queries = self.memory_hit_count + self.memory_miss_count
        hit_rate = 0.0
        if total_queries > 0:
            hit_rate = self.memory_hit_count.item() / total_queries.item()
            
        # Вычисляем среднее качество слияния
        avg_fusion_quality = 0.0
        if self.fusion_count > 0:
            avg_fusion_quality = self.fusion_quality_sum.item() / self.fusion_count.item()
            
        return {
            'memory_bank': memory_stats,
            'hit_rate': hit_rate,
            'memory_hit_count': self.memory_hit_count.item(),
            'memory_miss_count': self.memory_miss_count.item(),
            'avg_fusion_quality': avg_fusion_quality
        }


# Функция для создания модуля банка памяти
def create_memory_bank_module(config=None):
    """
    Создает модуль банка памяти на основе конфигурации.
    
    Args:
        config (dict, optional): Словарь с параметрами конфигурации
        
    Returns:
        MemoryBankModule: Модуль банка памяти
    """
    # Параметры по умолчанию
    default_config = {
        'feature_dim': 512,
        'color_channels': 2,
        'max_items': 100000,
        'index_type': 'flat',
        'gpu_id': -1,
        'save_dir': './data/memory_bank',
        'use_fusion': True,
        'k_neighbors': 5
    }
    
    # Объединяем с пользовательской конфигурацией
    if config is not None:
        for key, value in config.items():
            if key in default_config:
                default_config[key] = value
    
    # Создаем модель с указанными параметрами
    model = MemoryBankModule(
        feature_dim=default_config['feature_dim'],
        color_channels=default_config['color_channels'],
        max_items=default_config['max_items'],
        index_type=default_config['index_type'],
        gpu_id=default_config['gpu_id'],
        save_dir=default_config['save_dir'],
        use_fusion=default_config['use_fusion'],
        k_neighbors=default_config['k_neighbors']
    )
    
    return model


if __name__ == "__main__":
    # Пример использования модуля банка памяти
    
    # Создаем модуль
    memory_module = create_memory_bank_module({
        'feature_dim': 256,
        'max_items': 1000,
        'index_type': 'flat'
    })
    
    # Создаем тестовые данные
    batch_size = 2
    gray_image = torch.randn(batch_size, 1, 256, 256)
    color_image = torch.randn(batch_size, 2, 256, 256)  # ab каналы
    
    # Добавляем несколько элементов в банк памяти
    item_ids = memory_module.add_to_memory(gray_image, color_image)
    print(f"Добавлено элементов: {len(item_ids)}")
    
    # Колоризуем изображение с использованием банка памяти
    result = memory_module(gray_image)
    
    # Выводим информацию о результате
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Выводим статистики
    stats = memory_module.get_stats()
    print("\nСтатистики модуля:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")