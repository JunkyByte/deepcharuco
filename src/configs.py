import yaml
import cv2
from aruco_utils import get_board, board_image
from pydantic.dataclasses import dataclass


# HARDCODED for the time being
CONFIG_PATH = 'config.yaml'


@dataclass
class Config:
    board_name: str
    row_count: int
    col_count: int
    square_len: float
    marker_len: float

    input_size: tuple[int, int]
    num_workers: int
    bs_train: int
    bs_train_rn: int
    bs_val: int
    bs_val_rn: int
    train_labels: str
    val_labels: str
    train_images: str
    val_images: str

    # Self populated
    n_ids: int = None

    def __post_init__(self):
        self.n_ids = (self.row_count - 1) * (self.col_count - 1)


def load_configuration(path: str) -> Config:
    with open(path, 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return Config(**config_yaml)


if __name__ == '__main__':
    from aruco_utils import draw_inner_corners
    config = load_configuration(CONFIG_PATH)

    # Create an image from the gridboard
    img, corners = board_image(config.aruco_board, (480, 480),
                               config.row_count, config.col_count)
    img = draw_inner_corners(img, corners, np.arange(config.n_ids), draw_ids=True)
    cv2.imshow('Gridboard', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
