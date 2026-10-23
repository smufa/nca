/** Channel labels for the view mapping selects: `0 R … 3 A … 4 h4 …`. */

const CHANNEL_NAMES = ['R', 'G', 'B', 'A'];

export function channelLabel(i: number): string {
  return i < CHANNEL_NAMES.length ? `${i} ${CHANNEL_NAMES[i]}` : String(i);
}

export function fillChannelSelect(
  sel: HTMLSelectElement,
  channelN: number,
  includeOff: boolean,
): void {
  sel.innerHTML = '';
  if (includeOff) {
    const off = document.createElement('option');
    off.value = '-1';
    off.textContent = 'off';
    sel.appendChild(off);
  }
  for (let i = 0; i < channelN; i++) {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = channelLabel(i);
    sel.appendChild(opt);
  }
}
