import numpy as np
import os
import cv2
from sklearn.cluster import DBSCAN

# type alias
CV_Img = cv2.typing.MatLike
PointList = list[cv2.typing.Point]
PolygonList = list[PointList]
Areas = list[float]
Angle = float
Angles = list[Angle]
BBoxPropsList = tuple[PolygonList, PointList, Areas, Angles]


def int32_list(stuff) -> list:
    return np.array(stuff, dtype=np.int32).tolist()


def get_mean_deviation(
    angles: Angles, areas: Areas, DEBUG=True
) -> float:
    if DEBUG:
        print(f"angles: {angles}")

    # if just one element, return it
    if len(angles) == 1:
        return angles[0]
    elif len(angles) == 0:
        # if nothing, no rotation
        return 0
    data = np.array(angles).reshape(-1, 1)
    if DEBUG:
        print(f"reshaped: {data}")

    # eps = max distance within cluster, min_samples = minimum cluster size
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(data)

    labels = clustering.labels_
    unique_labels = [l for l in set(labels) if l != -1]  # -1 = noise

    if DEBUG:
        print(f"uniq labels: {unique_labels}")

    if len(unique_labels) == 0:
        # if all labels are considered as noise
        # fix for cases like doc_02275.png
        # use angle of bbox with largest area
        index = areas.index(max(areas))
        return angles[index]
    # find largest cluster
    largest_label = max(
        unique_labels, key=lambda l: np.sum(labels == l)
    )
    largest_cluster = data[labels == largest_label].flatten()
    if DEBUG:
        print(f"cluster: {largest_cluster}")
    return float(np.mean(largest_cluster))


def blur_and_invert(img: CV_Img) -> CV_Img:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    gray = cv2.bitwise_not(gray)
    return gray


def get_skew_params(
    gray_img: CV_Img,
) -> BBoxPropsList:
    thresh = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    kernel_size = int(gray_img.shape[1] * 0.025)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    dilate = cv2.dilate(thresh, kernel)
    contours, _ = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    img_area = gray_img.shape[0] * gray_img.shape[1]

    angles: Angles = []
    contour_bbox_coords: PolygonList = []
    contour_bbox_centers: PointList = []
    contour_bbox_area: Areas = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        center, dims, angle = rect
        rect_area = dims[0] * dims[1]
        center = int32_list(center)
        if rect_area >= img_area * 0.01:
            # only consider large patches of boxes
            box: PointList = int32_list(cv2.boxPoints(rect))
            contour_bbox_coords.append(box)
            contour_bbox_centers.append(center)
            contour_bbox_area.append(rect_area)
            angle = angle - 180 + 90
            if (90 - abs(angle)) < 1:
                angle = 0
            elif angle < -45:
                angle = angle + 90  # fix for cases like doc_03764.png
            angles.append(angle)
    return (
        contour_bbox_coords,
        contour_bbox_centers,
        contour_bbox_area,
        angles,
    )


def deskew(
    original: CV_Img,
) -> tuple[CV_Img, BBoxPropsList, Angle]:
    height: int = original.shape[0]
    width: int = original.shape[1]
    img = blur_and_invert(original)
    bbox_props = get_skew_params(img)
    _, _, contour_bbox_area, angles = bbox_props
    deskewed = original.copy()

    # TODO: way to optimimum angle

    angle = get_mean_deviation(angles, contour_bbox_area)
    print(f"rotating by {angle}")
    m = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    deskewed = cv2.warpAffine(
        deskewed, m, (width, height), borderValue=(255, 255, 255)
    )
    return (
        deskewed,
        bbox_props,
        angle,
    )


def annotate_skews(
    img: CV_Img,
    bbox_props: BBoxPropsList,
    rotation: Angle,
) -> CV_Img:
    img_ = img.copy()
    contour_bbox_coords, contour_bbox_centers, _, angles = bbox_props
    for box, center, ang in zip(
        contour_bbox_coords, contour_bbox_centers, angles
    ):
        box = np.array(box, dtype=np.int32).reshape((-1, 1, 2))

        # Rectangle
        img_ = cv2.polylines(
            img_,
            [box],
            isClosed=True,
            color=(0, 0, 255),
            thickness=1,
        )

        # Angle
        img_ = cv2.putText(
            img_,
            f"{ang:.1f}",
            center,
            fontFace=cv2.QT_FONT_NORMAL,
            fontScale=0.5,
            color=(0, 0, 255),
        )
    img_ = cv2.putText(
        img_,
        f"{rotation:.2f}",
        (10, 20),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(255, 0, 0),
    )
    return img_


def deskew_image(
    src_img_path: str,
    out_dir: str = "./out",
    save_annotated_img: bool = True,
    write_threshold: float = 0,
):
    """deskews image

    Args:
        src_img_path (str): relative path of image
        out_dir (str, optional): output directory. Defaults to "out".
        save_annotated_img (bool, optional): whether to save annotated bboxs in image. Defaults to True.
        write_threshold (float, optional): will only write if rotation angle is greater than this. Defaults to 0.
    """
    img_name = os.path.basename(src_img_path)
    out_path = os.path.join(out_dir, img_name)

    src_img = cv2.imread(src_img_path)
    if src_img is None:
        print(f"{src_img_path} not found")
        return

    deskewed, bbox_props, angle = deskew(src_img)
    if abs(angle) > write_threshold:
    #     cv2.imwrite(out_path, deskewed)
    #     if save_annotated_img:
    #         annoted_img = annotate_skews(
    #             src_img,
    #             bbox_props,
    #             angle,
    #         )
    #         (fn, ext) = os.path.splitext(img_name)
    #         cv2.imwrite(
    #             os.path.join(out_dir, fn + "_annot" + ext),
    #             annoted_img,
    #         )
        return deskewed
        
    else:
        print("threshold not met, skip writing")
        return None

