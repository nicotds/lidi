def test_importa_modulo():
    from src import cuantizacion as c
    for nombre in c.__all__:
        assert hasattr(c, nombre)