import warnings


def check_if_resizing_is_too_big(img_size, out_size):
    is_resizing_too_big = any(
        outdim < featdim / 1.5 or outdim > featdim * 1.5
        for outdim, featdim in zip(out_size, img_size)
    )

    if is_resizing_too_big:
        warnings.warn(
            f"""
            End Resizing Warning: Something might be wrong with the decoder.
            Output shape: {out_size} - Input shape: {img_size}
            """
        )
