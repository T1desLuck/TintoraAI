"""
FPN + Pyramid Pooling: Модуль для обеспечения мультимасштабности в колоризаторе.

Данный модуль объединяет Feature Pyramid Network (FPN) и Pyramid Pooling
для эффективного извлечения и обработки признаков на разных уровнях детализации.
Это позволяет модели лучше понимать как крупные объекты, так и мелкие детали,
что критично для качественной колоризации изображений.

Основные компоненты:
- Feature Pyramid Network: Иерархическая структура признаков разных масштабов
- Pyramid Pooling Module: Агрегация контекстной информации на разных масштабах
- Lateral Connections: Соединения между уровнями для обмена информацией
- Bottleneck Blocks: Эффективные блоки для преобразования признаков
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBNReLU(nn.Module):
    """
    Блок Conv-BatchNorm-ReLU для FPN.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        kernel_size (int): Размер ядра свертки
        stride (int): Шаг свертки
        padding (int): Отступ для свертки
        use_bn (bool): Использовать ли BatchNorm
        use_relu (bool): Использовать ли ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_bn=True, use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()
        
    def forward(self, x):
        """
        Прямое распространение.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W]
            
        Returns:
            torch.Tensor: Выходной тензор [B, out_channels, H', W']
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck блок для FPN.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        stride (int): Шаг свертки
        downsample (nn.Module, optional): Downsampling слой
        expansion (int): Коэффициент расширения каналов
        groups (int): Количество групп для GroupNorm
        width_per_group (int): Ширина каналов на группу
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, expansion=4,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        
        width = int(out_channels * (width_per_group / 64.)) * groups
        
        # 1x1 conv для снижения размерности
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        # 3x3 conv для обработки признаков
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, 
                              groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        # 1x1 conv для повышения размерности
        self.conv3 = nn.Conv2d(width, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        """
        Прямое распространение через Bottleneck.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W]
            
        Returns:
            torch.Tensor: Выходной тензор [B, out_channels*expansion, H', W']
        """
        identity = x
        
        # Первый свёрточный слой (1x1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Второй свёрточный слой (3x3)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Третий свёрточный слой (1x1)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module для агрегации контекстной информации.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        pool_sizes (list): Список размеров для пулинга
        use_bn (bool): Использовать ли BatchNorm
    """
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], use_bn=True):
        super(PyramidPoolingModule, self).__init__()
        
        self.pool_sizes = pool_sizes
        self.use_bn = use_bn
        inter_channels = in_channels // len(pool_sizes)  # Распределяем каналы равномерно
        
        # Создаем ветви для разных масштабов пулинга
        self.paths = nn.ModuleList()
        for pool_size in pool_sizes:
            self.paths.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBNReLU(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, use_bn=use_bn)
            ))
        
        # Финальный сверточный слой для объединения всех признаков
        self.fuse_conv = ConvBNReLU(
            in_channels + inter_channels * len(pool_sizes),
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn
        )
        
    def forward(self, x):
        """
        Прямое распространение через PPM.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W]
            
        Returns:
            torch.Tensor: Выходной тензор с агрегированным контекстом [B, out_channels, H, W]
        """
        size = x.size()
        feat_list = [x]
        
        # Применяем пулинг разных масштабов
        for path in self.paths:
            # Ветвь обработки
            feat = path[0](x)  # AdaptiveAvgPool2d
            feat = path[1](feat)  # ConvBNReLU
            # Билинейная интерполяция до исходного размера
            feat = F.interpolate(feat, size=size[2:], mode='bilinear', align_corners=True)
            feat_list.append(feat)
        
        # Конкатенация всех признаков
        feat = torch.cat(feat_list, dim=1)
        
        # Финальная свертка для объединения
        feat = self.fuse_conv(feat)
        
        return feat


class FPN(nn.Module):
    """
    Feature Pyramid Network для создания иерархии признаков.
    
    Args:
        in_channels_list (list): Список количества каналов для каждого уровня входа
        out_channels (int): Количество выходных каналов для каждого уровня FPN
        use_bn (bool): Использовать ли BatchNorm
    """
    def __init__(self, in_channels_list, out_channels, use_bn=True):
        super(FPN, self).__init__()
        
        # Боковые соединения (lateral connections) для каждого уровня
        self.lateral_convs = nn.ModuleList()
        # Выходные свертки для каждого уровня
        self.output_convs = nn.ModuleList()
        
        # Создаем боковые соединения и выходные свертки для каждого уровня
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_bn=use_bn)
            )
            self.output_convs.append(
                ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
            )
            
    def forward(self, inputs):
        """
        Прямое распространение через FPN.
        
        Args:
            inputs (list): Список тензоров признаков [B, C_i, H_i, W_i] для каждого уровня (от низшего к высшему)
            
        Returns:
            list: Список тензоров признаков FPN [B, out_channels, H_i, W_i] для каждого уровня
        """
        assert len(inputs) == len(self.lateral_convs), "Количество входов должно соответствовать количеству боковых соединений"
        
        # Список для хранения выходных признаков
        fpn_features = []
        
        # Обработка верхнего уровня (самый абстрактный)
        lateral_feat = self.lateral_convs[-1](inputs[-1])
        output_feat = self.output_convs[-1](lateral_feat)
        fpn_features.append(output_feat)
        
        # Обработка остальных уровней (снизу вверх)
        for i in range(len(inputs) - 2, -1, -1):
            # Боковое соединение
            lateral_feat = self.lateral_convs[i](inputs[i])
            
            # Верхний уровень (требуется upsampling)
            top_down_feat = F.interpolate(
                fpn_features[0], size=lateral_feat.shape[2:], mode='bilinear', align_corners=True
            )
            
            # Слияние признаков
            lateral_feat = lateral_feat + top_down_feat
            
            # Выходная свертка
            output_feat = self.output_convs[i](lateral_feat)
            
            # Вставляем текущий уровень в начало списка (сохраняя порядок от высших к низшим)
            fpn_features.insert(0, output_feat)
            
        return fpn_features


class AdaptiveFPN(nn.Module):
    """
    Адаптивный Feature Pyramid Network с взвешенным слиянием.
    
    Args:
        in_channels_list (list): Список количества каналов для каждого уровня входа
        out_channels (int): Количество выходных каналов для каждого уровня FPN
        use_bn (bool): Использовать ли BatchNorm
        use_attention (bool): Использовать ли механизм внимания для взвешивания
    """
    def __init__(self, in_channels_list, out_channels, use_bn=True, use_attention=True):
        super(AdaptiveFPN, self).__init__()
        
        # Боковые соединения (lateral connections) для каждого уровня
        self.lateral_convs = nn.ModuleList()
        # Выходные свертки для каждого уровня
        self.output_convs = nn.ModuleList()
        # Веса для адаптивного слияния (если используется attention)
        self.use_attention = use_attention
        
        if self.use_attention:
            self.attention_weights = nn.ModuleList()
            
        # Создаем боковые соединения и выходные свертки для каждого уровня
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_bn=use_bn)
            )
            self.output_convs.append(
                ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
            )
            
            if self.use_attention:
                self.attention_weights.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels // 16, 1, kernel_size=1),
                        nn.Sigmoid()
                    )
                )
            
    def forward(self, inputs):
        """
        Прямое распространение через адаптивный FPN.
        
        Args:
            inputs (list): Список тензоров признаков [B, C_i, H_i, W_i] для каждого уровня (от низшего к высшему)
            
        Returns:
            list: Список тензоров признаков FPN [B, out_channels, H_i, W_i] для каждого уровня
        """
        assert len(inputs) == len(self.lateral_convs), "Количество входов должно соответствовать количеству боковых соединений"
        
        # Боковые соединения (преобразование всех уровней в единый канал)
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        
        # Список для хранения выходных признаков
        fpn_features = []
        
        # Сначала обрабатываем верхний уровень
        fpn_features.append(self.output_convs[-1](laterals[-1]))
        
        # Затем обрабатываем остальные уровни (снизу вверх)
        for i in range(len(laterals) - 2, -1, -1):
            # Масштабирование верхнего уровня до размера текущего
            if self.use_attention:
                # Вычисление весов внимания
                attention_weight = self.attention_weights[i](laterals[i])
                
                # Апсемплинг признаков верхнего уровня
                upsampled_feat = F.interpolate(
                    fpn_features[0], size=laterals[i].shape[2:], mode='bilinear', align_corners=True
                )
                
                # Взвешенное слияние
                merged = laterals[i] + attention_weight * upsampled_feat
            else:
                # Обычное слияние без взвешивания
                upsampled_feat = F.interpolate(
                    fpn_features[0], size=laterals[i].shape[2:], mode='bilinear', align_corners=True
                )
                merged = laterals[i] + upsampled_feat
            
            # Выходная свертка
            output = self.output_convs[i](merged)
            fpn_features.insert(0, output)
            
        return fpn_features


class ScalePyramidPooling(nn.Module):
    """
    Расширенный Pyramid Pooling для захвата признаков на разных масштабах.
    
    Args:
        in_channels (int): Количество входных каналов
        out_channels (int): Количество выходных каналов
        scales (list): Список масштабов для пулинга (доли от входного размера)
        use_bn (bool): Использовать ли BatchNorm
    """
    def __init__(self, in_channels, out_channels, scales=[0.25, 0.5, 0.75, 1.0], use_bn=True):
        super(ScalePyramidPooling, self).__init__()
        
        self.scales = scales
        inter_channels = in_channels // 4
        
        # Создаем ветви для разных масштабов
        self.scale_convs = nn.ModuleList()
        for _ in scales:
            self.scale_convs.append(
                nn.Sequential(
                    ConvBNReLU(in_channels, inter_channels, kernel_size=1, padding=0, use_bn=use_bn),
                    ConvBNReLU(inter_channels, inter_channels, kernel_size=3, padding=1, use_bn=use_bn)
                )
            )
        
        # Финальная свертка для объединения всех признаков
        self.fusion_conv = ConvBNReLU(
            in_channels + inter_channels * len(scales),
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn
        )
        
    def forward(self, x):
        """
        Прямое распространение через Scale Pyramid Pooling.
        
        Args:
            x (torch.Tensor): Входной тензор [B, C, H, W]
            
        Returns:
            torch.Tensor: Выходной тензор с признаками разных масштабов [B, out_channels, H, W]
        """
        ori_size = x.size()[2:]
        feat_list = [x]
        
        # Обработка на разных масштабах
        for i, scale in enumerate(self.scales):
            if scale == 1.0:
                # Обрабатываем без изменения размера
                feat = self.scale_convs[i](x)
            else:
                # Изменяем размер, обрабатываем и возвращаем к исходному размеру
                feat_size = [int(s * scale) for s in ori_size]
                scaled = F.interpolate(x, size=feat_size, mode='bilinear', align_corners=True)
                feat = self.scale_convs[i](scaled)
                feat = F.interpolate(feat, size=ori_size, mode='bilinear', align_corners=True)
                
            feat_list.append(feat)
        
        # Конкатенация всех признаков
        feat = torch.cat(feat_list, dim=1)
        
        # Финальная свертка
        output = self.fusion_conv(feat)
        
        return output


class MultiLevelFusion(nn.Module):
    """
    Слияние признаков с разных уровней иерархии.
    
    Args:
        in_channels_list (list): Список количества каналов для каждого уровня
        out_channels (int): Количество выходных каналов
        use_bn (bool): Использовать ли BatchNorm
        use_attention (bool): Использовать ли механизм внимания
    """
    def __init__(self, in_channels_list, out_channels, use_bn=True, use_attention=True):
        super(MultiLevelFusion, self).__init__()
        
        self.use_attention = use_attention
        num_levels = len(in_channels_list)
        
        # Выравнивание размерностей для каждого уровня
        self.align_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.align_convs.append(
                ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0, use_bn=use_bn)
            )
            
        # Механизм внимания для взвешивания каждого уровня
        if use_attention:
            self.attention_weights = nn.ModuleList()
            for _ in range(num_levels):
                self.attention_weights.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels // 16, 1, kernel_size=1),
                        nn.Sigmoid()
                    )
                )
                
        # Финальная свертка для объединения
        self.fusion_conv = ConvBNReLU(
            out_channels * num_levels if not use_attention else out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn
        )
            
    def forward(self, inputs):
        """
        Прямое распространение через Multi-Level Fusion.
        
        Args:
            inputs (list): Список тензоров признаков [B, C_i, H_i, W_i] для каждого уровня
            
        Returns:
            torch.Tensor: Слитые признаки [B, out_channels, H_0, W_0]
        """
        assert len(inputs) == len(self.align_convs), "Количество входов должно соответствовать количеству слоев выравнивания"
        
        # Целевой размер (размер самого детального уровня)
        target_size = inputs[0].shape[2:]
        
        # Выравнивание размерностей каналов
        aligned_features = [conv(x) for x, conv in zip(inputs, self.align_convs)]
        
        # Приведение всех уровней к целевому размеру
        for i in range(1, len(aligned_features)):
            aligned_features[i] = F.interpolate(
                aligned_features[i], size=target_size, mode='bilinear', align_corners=True
            )
            
        if self.use_attention:
            # Взвешенное слияние с вниманием
            weighted_sum = torch.zeros_like(aligned_features[0])
            attention_maps = []
            
            for i, feat in enumerate(aligned_features):
                weight = self.attention_weights[i](feat)
                attention_maps.append(weight)
                weighted_sum += feat * weight
                
            # Нормализация весов (необязательно, т.к. sigmoid уже ограничивает значения)
            fused = self.fusion_conv(weighted_sum)
            
            # Возвращаем и веса внимания для визуализации
            return fused, attention_maps
        else:
            # Простая конкатенация
            fused = torch.cat(aligned_features, dim=1)
            fused = self.fusion_conv(fused)
            return fused, None


class FPNPyramid(nn.Module):
    """
    Полная реализация FPN с Pyramid Pooling для колоризации.
    
    Args:
        backbone_channels (list): Список каналов из backbone на разных уровнях
        fpn_channels (int): Количество каналов для FPN
        output_channels (int): Количество каналов на выходе
        use_ppm (bool): Использовать ли Pyramid Pooling Module
        output_stride (int): Шаг выходного слоя относительно входа
        use_attention (bool): Использовать ли механизм внимания
    """
    def __init__(self, backbone_channels, fpn_channels=256, output_channels=128,
                 use_ppm=True, output_stride=4, use_attention=True):
        super(FPNPyramid, self).__init__()
        
        # FPN для иерархии признаков
        self.fpn = AdaptiveFPN(backbone_channels, fpn_channels, use_bn=True, use_attention=use_attention)
        
        # Pyramid Pooling Module (если используется)
        self.use_ppm = use_ppm
        if use_ppm:
            self.ppm = PyramidPoolingModule(
                fpn_channels, 
                fpn_channels, 
                pool_sizes=[1, 2, 3, 6], 
                use_bn=True
            )
            
        # Scale Pyramid Pooling для низкоуровневых признаков
        self.spp = ScalePyramidPooling(
            fpn_channels, 
            fpn_channels, 
            scales=[0.25, 0.5, 0.75, 1.0], 
            use_bn=True
        )
        
        # Multi-Level Fusion для объединения признаков разных уровней
        self.fusion = MultiLevelFusion(
            [fpn_channels] * len(backbone_channels),
            fpn_channels,
            use_bn=True,
            use_attention=use_attention
        )
        
        # Выходной слой с возможностью апсемплинга
        self.output_conv = nn.Sequential(
            ConvBNReLU(fpn_channels, output_channels, kernel_size=3, padding=1, use_bn=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1)
        )
        
        self.output_stride = output_stride
        if output_stride < 4:  # FPN обычно выдает признаки с шагом 4
            self.upsample = nn.Upsample(
                scale_factor=4 // output_stride,
                mode='bilinear',
                align_corners=True
            )
        else:
            self.upsample = nn.Identity()
            
    def forward(self, backbone_features):
        """
        Прямое распространение через FPN с Pyramid Pooling.
        
        Args:
            backbone_features (list): Признаки из backbone на разных уровнях
            
        Returns:
            dict: {
                'fpn_features': list,  # Список признаков для каждого уровня FPN
                'ppm_features': torch.Tensor,  # Признаки после PPM (если используется)
                'spp_features': torch.Tensor,  # Признаки после SPP
                'fused_features': torch.Tensor,  # Объединенные признаки всех уровней
                'output': torch.Tensor  # Финальный выход
            }
        """
        # Получаем признаки FPN
        fpn_features = self.fpn(backbone_features)
        
        # Результаты для возврата
        results = {'fpn_features': fpn_features}
        
        # Применяем PPM к верхнему (самому абстрактному) уровню FPN, если используется
        if self.use_ppm:
            ppm_features = self.ppm(fpn_features[-1])
            results['ppm_features'] = ppm_features
            # Заменяем верхний уровень FPN на признаки после PPM
            fpn_features[-1] = ppm_features
            
        # Применяем SPP к нижнему (самому детальному) уровню FPN
        spp_features = self.spp(fpn_features[0])
        results['spp_features'] = spp_features
        # Заменяем нижний уровень FPN на признаки после SPP
        fpn_features[0] = spp_features
        
        # Multi-Level Fusion для объединения всех уровней
        fused_features, attention_maps = self.fusion(fpn_features)
        results['fused_features'] = fused_features
        if attention_maps:
            results['attention_maps'] = attention_maps
            
        # Финальная свертка и апсемплинг (если нужен)
        output = self.output_conv(fused_features)
        output = self.upsample(output)
        results['output'] = output
        
        return results


class AsymmetricFPN(nn.Module):
    """
    Асимметричный FPN с разным количеством каналов для разных уровней.
    Лучше адаптирован для колоризации, где низкоуровневые детали критичны.
    
    Args:
        in_channels_list (list): Список количества каналов для каждого уровня входа
        out_channels_list (list): Список количества выходных каналов для каждого уровня
        use_bn (bool): Использовать ли BatchNorm
    """
    def __init__(self, in_channels_list, out_channels_list, use_bn=True):
        super(AsymmetricFPN, self).__init__()
        
        assert len(in_channels_list) == len(out_channels_list), "Длина списков входных и выходных каналов должна совпадать"
        
        # Боковые соединения для каждого уровня
        self.lateral_convs = nn.ModuleList()
        # Выходные свертки для каждого уровня
        self.output_convs = nn.ModuleList()
        # Upsampling свертки для связи между уровнями
        self.upsampling_convs = nn.ModuleList()
        
        # Создаем слои для каждого уровня
        for i in range(len(in_channels_list)):
            # Боковые соединения
            self.lateral_convs.append(
                ConvBNReLU(in_channels_list[i], out_channels_list[i], kernel_size=1, padding=0, use_bn=use_bn)
            )
            
            # Выходные свертки
            self.output_convs.append(
                ConvBNReLU(out_channels_list[i], out_channels_list[i], kernel_size=3, padding=1, use_bn=use_bn)
            )
            
            # Upsampling свертки (кроме самого верхнего уровня)
            if i < len(in_channels_list) - 1:
                self.upsampling_convs.append(
                    ConvBNReLU(
                        out_channels_list[i+1], 
                        out_channels_list[i], 
                        kernel_size=1, padding=0, use_bn=use_bn
                    )
                )
                
    def forward(self, inputs):
        """
        Прямое распространение через асимметричный FPN.
        
        Args:
            inputs (list): Список тензоров признаков [B, C_i, H_i, W_i] для каждого уровня (от низшего к высшему)
            
        Returns:
            list: Список тензоров признаков FPN [B, out_channels_i, H_i, W_i] для каждого уровня
        """
        assert len(inputs) == len(self.lateral_convs), "Количество входов должно соответствовать количеству боковых соединений"
        
        # Боковые соединения (преобразование входов)
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        
        # Список для хранения выходных признаков FPN
        fpn_features = [None] * len(laterals)
        
        # Обрабатываем верхний уровень
        fpn_features[-1] = self.output_convs[-1](laterals[-1])
        
        # Обрабатываем остальные уровни (сверху вниз)
        for i in range(len(laterals) - 2, -1, -1):
            # Upsampling и преобразование верхнего уровня
            upsampled = F.interpolate(
                fpn_features[i+1], size=laterals[i].shape[2:], mode='bilinear', align_corners=True
            )
            upsampled = self.upsampling_convs[i](upsampled)
            
            # Слияние с текущим уровнем
            merged = laterals[i] + upsampled
            
            # Выходная свертка
            fpn_features[i] = self.output_convs[i](merged)
            
        return fpn_features


class DynamicFPNPyramid(nn.Module):
    """
    Динамический FPN Pyramid с адаптивным распределением внимания между уровнями.
    
    Args:
        backbone_channels (list): Список каналов из backbone на разных уровнях
        fpn_channels (int): Количество каналов для FPN
        output_channels (int): Количество каналов на выходе
        use_context_module (bool): Использовать ли контекстный модуль (PPM или ASPP)
        context_module_type (str): Тип контекстного модуля ('ppm' или 'aspp')
    """
    def __init__(self, backbone_channels, fpn_channels=256, output_channels=128,
                 use_context_module=True, context_module_type='ppm'):
        super(DynamicFPNPyramid, self).__init__()
        
        # Настройка каналов для асимметричного FPN (больше каналов для низких уровней)
        channels_ratio = [1.5, 1.25, 1.0, 0.75]  # Пример соотношения каналов
        out_channels_list = [int(fpn_channels * r) for r in channels_ratio[:len(backbone_channels)]]
        
        # Асимметричный FPN
        self.fpn = AsymmetricFPN(backbone_channels, out_channels_list, use_bn=True)
        
        # Контекстный модуль (если используется)
        self.use_context_module = use_context_module
        if use_context_module:
            if context_module_type == 'ppm':
                self.context_module = PyramidPoolingModule(
                    out_channels_list[-1], 
                    out_channels_list[-1],
                    pool_sizes=[1, 2, 3, 6],
                    use_bn=True
                )
            else:  # 'aspp'
                # Можно реализовать ASPP (Atrous Spatial Pyramid Pooling) как альтернативу
                self.context_module = ScalePyramidPooling(
                    out_channels_list[-1],
                    out_channels_list[-1],
                    scales=[0.25, 0.5, 0.75, 1.0],
                    use_bn=True
                )
        
        # Динамическое распределение внимания между уровнями
        self.dynamic_attention = nn.ModuleList()
        for ch in out_channels_list:
            self.dynamic_attention.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(ch, ch // 16, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // 16, 1, kernel_size=1),
                    nn.Sigmoid()
                )
            )
            
        # Слияние всех уровней
        self.fusion = ConvBNReLU(
            sum(out_channels_list), 
            fpn_channels,
            kernel_size=3,
            padding=1,
            use_bn=True
        )
        
        # Выходной слой
        self.output_conv = nn.Sequential(
            ConvBNReLU(fpn_channels, output_channels, kernel_size=3, padding=1, use_bn=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1)
        )
        
    def forward(self, backbone_features):
        """
        Прямое распространение через динамический FPN Pyramid.
        
        Args:
            backbone_features (list): Признаки из backbone на разных уровнях
            
        Returns:
            dict: {
                'fpn_features': list,  # Список признаков FPN
                'attention_maps': list,  # Карты внимания для каждого уровня
                'fused_features': torch.Tensor,  # Объединенные признаки
                'output': torch.Tensor  # Финальный выход
            }
        """
        # Получаем признаки FPN
        fpn_features = self.fpn(backbone_features)
        
        # Применяем контекстный модуль к верхнему уровню (если используется)
        if self.use_context_module:
            context_features = self.context_module(fpn_features[-1])
            fpn_features[-1] = context_features
            
        # Применяем динамическое внимание к каждому уровню
        attention_weighted_features = []
        attention_maps = []
        
        # Целевой размер (размер самого детального уровня)
        target_size = fpn_features[0].shape[2:]
        
        for i, feat in enumerate(fpn_features):
            # Вычисляем веса внимания
            attention = self.dynamic_attention[i](feat)
            attention_maps.append(attention)
            
            # Применяем веса
            weighted_feat = feat * attention
            
            # Апсемплим до целевого размера (если нужно)
            if i > 0:
                weighted_feat = F.interpolate(
                    weighted_feat, size=target_size, mode='bilinear', align_corners=True
                )
                
            attention_weighted_features.append(weighted_feat)
            
        # Объединяем все взвешенные признаки
        fused = torch.cat(attention_weighted_features, dim=1)
        fused = self.fusion(fused)
        
        # Финальный выходной слой
        output = self.output_conv(fused)
        
        return {
            'fpn_features': fpn_features,
            'attention_maps': attention_maps,
            'fused_features': fused,
            'output': output
        }


def create_fpn_pyramid(backbone_channels=[256, 512, 1024, 2048], fpn_channels=256, output_channels=128):
    """
    Создает модуль FPN с Pyramid Pooling для колоризации.
    
    Args:
        backbone_channels (list): Список каналов из backbone на разных уровнях
        fpn_channels (int): Количество каналов для FPN
        output_channels (int): Количество выходных каналов
        
    Returns:
        FPNPyramid: Модуль FPN с Pyramid Pooling
    """
    return FPNPyramid(
        backbone_channels=backbone_channels,
        fpn_channels=fpn_channels,
        output_channels=output_channels,
        use_ppm=True,
        output_stride=4,
        use_attention=True
    )


def create_dynamic_fpn_pyramid(backbone_channels=[256, 512, 1024, 2048], fpn_channels=256, output_channels=128):
    """
    Создает модуль Dynamic FPN Pyramid для колоризации с адаптивным распределением внимания.
    
    Args:
        backbone_channels (list): Список каналов из backbone на разных уровнях
        fpn_channels (int): Количество каналов для FPN
        output_channels (int): Количество выходных каналов
        
    Returns:
        DynamicFPNPyramid: Модуль Dynamic FPN Pyramid
    """
    return DynamicFPNPyramid(
        backbone_channels=backbone_channels,
        fpn_channels=fpn_channels,
        output_channels=output_channels,
        use_context_module=True,
        context_module_type='ppm'
    )


# Пример использования
if __name__ == "__main__":
    # Создаем модель
    model = create_fpn_pyramid()
    
    # Создаем входные данные (имитация признаков из backbone)
    batch_size = 2
    input_features = [
        torch.randn(batch_size, 256, 64, 64),    # C2 (1/4 разрешения)
        torch.randn(batch_size, 512, 32, 32),    # C3 (1/8 разрешения)
        torch.randn(batch_size, 1024, 16, 16),   # C4 (1/16 разрешения)
        torch.randn(batch_size, 2048, 8, 8)      # C5 (1/32 разрешения)
    ]
    
    # Прямое распространение
    output_dict = model(input_features)
    
    # Вывод информации о формах выходных тензоров
    for k, v in output_dict.items():
        if isinstance(v, list):
            print(f"{k}: {[tensor.shape for tensor in v]}")
        elif isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
            
    print("\nТестирование модели Dynamic FPN Pyramid:")
    dynamic_model = create_dynamic_fpn_pyramid()
    dynamic_output = dynamic_model(input_features)
    
    for k, v in dynamic_output.items():
        if isinstance(v, list):
            print(f"{k}: {[tensor.shape for tensor in v]}")
        elif isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")