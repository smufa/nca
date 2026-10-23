/**
 * The playground's entry point: wire the controls and start the loop.
 *
 * The UI lives in `ui/`; this file exists so the wiring order is visible in one
 * place. `ui/app.ts` holds the shared state, `ui/loop.ts` the frame loop, and each
 * panel module wires its own controls.
 */

import './style.css';
import { app, initialEp } from './ui/app';
import { wireBrushControls } from './ui/brush';
import { wireCondControls } from './ui/cond';
import { installConsoleHelpers } from './ui/console';
import { wireKernelControls } from './ui/kernels';
import { start, wireControls } from './ui/loop';

app.ep = initialEp();

wireBrushControls();
wireKernelControls();
wireCondControls();
wireControls();
installConsoleHelpers();
start();
