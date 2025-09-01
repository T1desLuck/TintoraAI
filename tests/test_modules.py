import torch

# Импорт компонентов модели
from src.models.backbone import ConvNeXtTiny, CoAtNetLight
from src.models.backbone.gat_light import GATLight
from src.models.heads.heads import DepthHead, IlluminationHead
from src.models.crb.crb import ColorReasoningBlock
from src.models.decoder import UNetPPDecoder

# Тесты бэкоуна (Backbone)


def test_convnext_tiny():
    model = ConvNeXtTiny(in_channels=1)
    input_tensor = torch.randn(1, 1, 224, 224)
    output = model(input_tensor)
    # Проверяем базовые инварианты: список фич, правильная spatial кратность, разумное число каналов
    assert isinstance(output, (list, tuple)) and len(output) >= 1
    o0 = output[0]
    assert o0.shape[-2:] == (56, 56)  # H/4, W/4 для входа 224
    assert (
        o0.shape[1] >= 64
    )  # число каналов зависит от timm, но не должно быть слишком мало


def test_coatnet_light():
    # В TintoraAI после ConvNeXt первый CoAtNet получает тензор c1=96 каналов
    model = CoAtNetLight(in_channels=96)
    input_tensor = torch.randn(1, 96, 56, 56)  # F1
    feats = model(input_tensor)
    # Ожидаем как минимум 3 стадии и прогрессивное уменьшение разрешения
    assert isinstance(feats, (list, tuple)) and len(feats) >= 3
    h0, w0 = feats[0].shape[-2:]
    h1, w1 = feats[1].shape[-2:]
    h2, w2 = feats[2].shape[-2:]
    assert h0 >= h1 >= h2 and w0 >= w1 >= w2


def test_gat_light():
    model = GATLight()
    x = torch.randn(1, 192, 32, 32)
    output = model(x)
    assert output[-1].shape == (1, 384, 16, 16), "Неверная форма выхода GAT-light"


# Тесты голов (Heads)


def test_depth_head():
    # Согласуем каналы с моделью: c2=192, c3=384
    model = DepthHead(c2=192, c3=384)
    f2 = torch.randn(1, 192, 28, 28)
    f3 = torch.randn(1, 384, 16, 16)
    output = model(f2, f3, out_hw=(224, 224))
    assert output.shape[1] == 1, "Неверная форма выхода DepthHead"


def test_illumination_head():
    # Согласуем каналы с моделью: c2=192, c3=384
    model = IlluminationHead(c2=192, c3=384)
    f2 = torch.randn(1, 192, 28, 28)
    f3 = torch.randn(1, 384, 16, 16)
    output = model(f2, f3, out_hw=(224, 224))
    assert output.shape[1] == 1, "Неверная форма выхода IlluminationHead"


# Тест OMM

# def test_omm():
#     omm = ObjectMemoryModule(dim=256)
#     F2 = torch.randn(1, 192, 32, 32)
#     F3 = torch.randn(1, 384, 16, 16)
#     c_obj, color_hint = omm(F2, F3)
#     assert c_obj.shape == (1, 256), "OMM c_obj output shape is incorrect"
#     assert color_hint.shape == (1, 64, 2), "OMM color_hint output shape is incorrect"

# Тест CRB


def test_crb():
    crb = ColorReasoningBlock(c3=384, cmem=256)
    f3 = torch.randn(1, 384, 16, 16)  # B, C3, H/16, W/16
    mem_map = torch.randn(1, 256)  # B, C_mem
    d_map = torch.randn(1, 1, 16, 16)  # Приводим к разрешению F3 для конкатенации
    i_map = torch.randn(1, 1, 16, 16)
    normals = torch.randn(1, 3, 16, 16)  # B, 3, H/16, W/16
    # mem_map должен быть картой (B,C,H,W); приведём к размеру входа F3
    mem_map = torch.randn(1, 256, 16, 16)
    film = crb(F3=f3, mem_map=mem_map, D=d_map, I=i_map, normals=normals)
    # Ожидаются списки gamma/beta для трёх стадий
    assert set(film.keys()) == {"gamma", "beta"}
    assert len(film["gamma"]) == len(film["beta"]) == 3
    assert film["gamma"][0].shape[0] == 1 and film["beta"][0].shape[0] == 1


# Тест декодера


def test_decoder_unetpp():
    decoder = UNetPPDecoder(c1=96, c2=192, c3=384)
    f1 = torch.randn(1, 96, 64, 64)
    f2 = torch.randn(1, 192, 28, 28)
    f3 = torch.randn(1, 384, 16, 16)
    c_color = torch.randn(1, 512)

    # Эмулируем генерацию FiLM‑параметров из c_color, как ожидает forward декодера.
    # Декодер ожидает раздельные векторы gamma/beta для каждого уровня признаков.
    c_dims = {"c1": 96, "c2": 192, "c3": 384}
    gamma_gen_c3 = torch.nn.Linear(512, c_dims["c3"])
    beta_gen_c3 = torch.nn.Linear(512, c_dims["c3"])
    gamma_gen_c2 = torch.nn.Linear(512, c_dims["c2"])
    beta_gen_c2 = torch.nn.Linear(512, c_dims["c2"])
    gamma_gen_c1 = torch.nn.Linear(512, c_dims["c1"])
    beta_gen_c1 = torch.nn.Linear(512, c_dims["c1"])

    gammas = [gamma_gen_c3(c_color), gamma_gen_c2(c_color), gamma_gen_c1(c_color)]
    betas = [beta_gen_c3(c_color), beta_gen_c2(c_color), beta_gen_c1(c_color)]

    ab, sat = decoder(f1, f2, f3, gammas, betas, out_size=(256, 256))
    assert ab.shape[1] == 2, "Неверная форма выхода декодера: каналов ab должно быть 2"
    assert (
        sat.shape[1] == 1
    ), "Неверная форма выхода декодера: каналов saturation должно быть 1"
