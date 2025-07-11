def count_boxes(result, conf_threshold=0.25):
    """
    Count the number of bounding boxes in the result with confidence above threshold.
    :param result: A single ultralytics Result object (not a list)
    :param conf_threshold: Confidence threshold
    :return: int - number of detected boxes above threshold
    """
    boxes = result.boxes

    if boxes is None or boxes.conf is None:
        print("No boxes or confidences found.")
        return 0

    confs = boxes.conf.cpu().numpy()
    filtered = confs > conf_threshold
    count = filtered.sum()

    # print(f"All confs: {confs}")
    # print(f"Filtered: {filtered}")
    print(f"Count: {count}")

    return int(count)
