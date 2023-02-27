import yaml
import cv2
from aruco_utils import get_board
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

    # Self populated
    aruco_board = None  # TODO? Whath's the class
    aruco_dict = None
    n_ids: int = None

    def __post_init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(getattr(cv2.aruco, self.board_name))
        self.aruco_board = get_board(self)
        self.n_ids = (self.row_count - 1) * (self.col_count - 1)


def load_configuration(path: str) -> Config:
    with open(path, 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return Config(**config_yaml)


if __name__ == '__main__':
    from aruco_utils import board_image, draw_inner_corners
    configs = load_configuration(CONFIG_PATH)

    # Create an image from the gridboard
    img, corners = board_image(configs.aruco_board, (480, 480),
                               configs.row_count, configs.col_count)
    img = draw_inner_corners(img, corners, draw_ids=True)
    cv2.imshow('Gridboard', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
