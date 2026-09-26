const POPUP_WIDTH = 360;
const GAP = 8;

export interface PopupPosition {
  left: number;
  width: number;
  maxHeight: number;
  /** Anchored by its bottom edge when placed above the marker, by its top edge when below. */
  bottom?: number;
  top?: number;
}

/**
 * Pins a popup to the viewport rather than to its marker's containing block.
 * A marker near the right edge or at the top of a long answer would otherwise
 * push the popup off screen, where it gets clipped instead of scrolled.
 */
export function computePopupPosition(marker: DOMRect): PopupPosition {
  const width = Math.min(POPUP_WIDTH, window.innerWidth - GAP * 2);
  const left = Math.min(Math.max(marker.left, GAP), window.innerWidth - width - GAP);

  const spaceAbove = marker.top - GAP * 2;
  const spaceBelow = window.innerHeight - marker.bottom - GAP * 2;

  if (spaceAbove >= spaceBelow) {
    return { left, width, maxHeight: spaceAbove, bottom: window.innerHeight - marker.top + GAP };
  }
  return { left, width, maxHeight: spaceBelow, top: marker.bottom + GAP };
}
