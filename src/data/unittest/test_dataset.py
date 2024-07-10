from src.data.pascalvoc2012 import VocDataset
from src.utils.visualization import Visualizer
from src.utils.data_utils import DataUtils
from src import config as cfg
from tqdm import tqdm


def test_vocdataset():
    dataset = VocDataset(mode='Train')
    for i in tqdm(range(10)):
        image, mask = dataset[i]
        image = DataUtils.image_to_numpy(image)
        # image = DataUtils.image_to_numpy(mask)
        Visualizer.save_debug(image, cfg['Debug']['dataset'], f'{i}.png')


test_vocdataset()