import numpy as np
import cv2
from models.model_utils import label_to_keypoints


def create_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def _detect_markers(gray: np.ndarray, dictionary, parameters):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def get_board_object_points(board) -> np.ndarray:
    if hasattr(board, "getObjPoints"):
        return np.array(board.getObjPoints(), dtype=np.float32)
    return np.array(board.objPoints, dtype=np.float32)


def cv2_aruco_detect(image, dictionary, board, parameters):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = _detect_markers(gray, dictionary, parameters)
    if hasattr(cv2.aruco, "refineDetectedMarkers"):
        try:
            corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                image, board, corners, ids, rejected
            )
        except TypeError:
            corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                image, board, corners, ids, np.array([])
            )

    # If markers are detected, draw them and the board inner corners
    if ids is not None and len(corners) > 0:
        # Get board corners
        board_corners = [corners[i][0] for i in range(len(corners))]
        board_corners = np.array(board_corners, dtype=np.float32)

        # Draw board inner corners
        image = draw_inner_corners(image, board_corners.reshape((-1, 2)), ids=np.arange(board_corners.shape[0]))

    return image, corners, ids


def get_board(configs):
    dictionary = get_aruco_dict(configs.board_name)

    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard(
                (configs.col_count, configs.row_count),
                configs.square_len,
                configs.marker_len,
                dictionary
            )
        except TypeError:
            pass

    return cv2.aruco.CharucoBoard_create(
        squaresX=configs.col_count,
        squaresY=configs.row_count,
        squareLength=configs.square_len,
        markerLength=configs.marker_len,
        dictionary=dictionary
    )


def get_aruco_dict(board_name):
    dict_id = getattr(cv2.aruco, board_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def board_image(board, resolution: tuple[int, int],
                row_count: int, col_count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a board image of given resolution and return the inner corners pixel position.

    Parameters
    ----------
    board : chess.Board
        The chess board object to create an image of.
    resolution : tuple of int
        A tuple containing the desired width and height of the output image.
    row_count : int
        The number of rows on the chessboard.
    col_count : int
        The number of columns on the chessboard.

    Returns
    -------
    tuple of ndarray
        A tuple containing the board image as a ndarray and an array of shape
        (N,2) representing the x and y coordinates of the inner corners of the
        chessboard, where N is (row_count-1)*(col_count-1).
        
    Notes
    -----
    This function assumes that the chess board object is a valid input and has
    been initialized properly.

    The inner corners pixel positions are calculated using the input parameters
    and returned as an array of shape (N,2), where each row corresponds to the
    pixel position of an inner corner in (x, y) format. The inner corners are
    defined as the intersections of the grid lines on the chessboard, excluding
    the outermost lines.

    The output image is generated using the `draw` method of the chess board
    object, which is converted to color format using `cv2.cvtColor` with the
    `cv2.COLOR_GRAY2BGR` flag.
    """
    if hasattr(board, "generateImage"):
        board_gray = board.generateImage(resolution)
    else:
        board_gray = board.draw(outSize=resolution)
    img = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
    pixel_offset = np.array([resolution[0] / col_count, resolution[1] / row_count])

    # row_id, col_id, (x, y) pixel coords
    inn_rc = np.arange(1, row_count)
    inn_cc = np.arange(1, col_count)
    corners = np.array(np.meshgrid(inn_rc, inn_cc)).reshape((2, -1)).T * pixel_offset
    return img, corners.astype(int)


def draw_inner_corners(img: np.ndarray, corners: np.ndarray, ids: np.ndarray,
                       draw_ids=False, radius=2, color=(0, 0, 255)) -> np.ndarray:
    """
    Draw circles on an input image at the specified corners' locations.

    Parameters
    ----------
    img : ndarray
        The input image on which corners will be drawn.
    corners : ndarray
        An array of shape (N, 2) representing the x and y coordinates of each
        corner.
    ids : ndarray
        An array of shape (N,) representing the id of each corner.
    draw_ids : bool, optional
        If True, the function will draw the id number next to each corner.
        Default is False.
    radius : int, optional
        The radius of the circle to be drawn at each corner. Default is 2.
    color : tuple, optional
        The color of the circles to be drawn. Default is (0,0,255), which
        corresponds to red in OpenCV's BGR color format.

    Returns
    -------
    ndarray
        The input image with the circles drawn at the specified corners'
        locations.

    Notes
    -----
    This function assumes that the input image is in color (i.e., has three
    channels).

    The function checks whether each corner's location is within the bounds of
    the input image before drawing the circle. If the location is outside the
    image boundaries, the circle will not be drawn.

    """
    assert img.ndim == 3 and img.shape[-1] == 3
    img = img.copy()

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text_thickness = 1

    for corner, idx in zip(corners, ids):
        corner = np.round(corner).astype(int)

        if corner[0] > img.shape[1] or corner[1] > img.shape[0]:
            continue
        cv2.circle(img, tuple(corner), radius=radius, color=color,
                   thickness=text_thickness)

        if draw_ids:
            label_size, _ = cv2.getTextSize(str(idx), font, .5, text_thickness)
            pos = (corner[0] - label_size[0] // 2 - 7, corner[1] + label_size[1] // 2 - 3)
            cv2.putText(img, str(idx), pos, font, .45, (0, 255, 0), text_thickness)
    return img


def draw_circle_pred(img: np.ndarray, loc: np.ndarray, ids: np.ndarray,
                     dust_bin_ids: int, draw_ids=False,
                     radius=2, color=(255, 0, 0)):
    """
    Draw circles on an input image at the locations of predicted keypoints and return the modified image.

    Parameters
    ----------
    img : ndarray
        A numpy array representing the input image.
    loc : ndarray
        The loc predicted by DeepCharuco
    ids : ndarray
        The ids predicted by DeepCharuco
    dust_bin_ids : int
        An integer value representing the id of the dustbin
    draw_ids : bool
        A boolean indicating whether or not to draw the ids of the keypoints.
    radius : int
        An integer value representing the radius of the circles to be drawn.
    color : tuple of int
        A tuple containing three integer values representing the RGB color code
        of the circles to be drawn.

    Returns
    -------
    ndarray
        A numpy array representing the modified image with circles drawn at the
        predicted keypoint locations.

    Notes
    -----
    This function assumes that the input image is a valid numpy array and has
    been initialized properly.

    The predicted keypoint locations and ids are used to draw circles on the
    input image. The dustbin keypoints are excluded from the output image, as
    specified by the `dust_bin_ids` parameter.

    If the `draw_ids` parameter is set to True, the corresponding ids for each
    predicted keypoint are also drawn on the image.

    The output image is a modified version of the input image with circles
    drawn at the predicted keypoint locations, and is returned as a numpy
    array.

    """
    assert loc.ndim == 2 and ids.ndim == 2
    img = img.copy()

    kpts, ids = label_to_keypoints(loc[None, ...], ids[None, ...], dust_bin_ids)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    for corner, ith in zip(kpts, ids):
        cv2.circle(img, corner, radius=radius, color=color,
                   thickness=text_thickness)

        if draw_ids:
            label_size, _ = cv2.getTextSize(str(ith), font, .5, text_thickness)
            pos = (corner[0] - label_size[0] // 2 + 3, corner[1] + label_size[1] // 2 + 3)
            cv2.putText(img, str(ith), pos, font, .3, (0, 255, 255), text_thickness)
    return img
