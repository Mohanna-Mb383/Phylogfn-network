"""Optimizer-update monitor with fail-fast (copied into <repo>/src/utils/update_monitor.py by apply_fix.py).

An update counts as ACCEPTED iff the Adam step counter advanced during generator.update_model(). This is independent
of how (or whether) the training code guards non-finite gradients, so it also catches silent skips.

Environment variables (no config-schema change needed, so it works with the historical fork's defaults.py):
  PHYLOGFN_MAX_CONSEC_REJECT      consecutive rejected updates before aborting       (default 50; 0 disables)
  PHYLOGFN_MAX_EPOCH_REJECT_FRAC  max fraction of rejected updates in one epoch      (default 0.25; 1 disables)
"""
import json
import math
import os
import time


def adam_step(opt):
    steps = [float(s['step']) for s in opt.state.values() if isinstance(s, dict) and 'step' in s]
    return int(max(steps)) if steps else 0


class TrainingFrozenError(RuntimeError):
    pass


class UpdateMonitor(object):
    def __init__(self, generator, output_path=None, max_consec=None, max_epoch_frac=None):
        self.generator = generator
        self.output_path = output_path
        self.max_consec = int(os.environ.get('PHYLOGFN_MAX_CONSEC_REJECT', 50) if max_consec is None else max_consec)
        self.max_frac = float(os.environ.get('PHYLOGFN_MAX_EPOCH_REJECT_FRAC', 0.25)
                              if max_epoch_frac is None else max_epoch_frac)
        self.consec = 0
        self.accepted = 0
        self.rejected = 0
        self.last_step = adam_step(generator.opt)
        self.events = []

    def after_update(self, epoch, t, info=None):
        step = adam_step(self.generator.opt)
        ok = step > self.last_step
        self.last_step = step
        if ok:
            self.accepted += 1
            self.consec = 0
            return True
        self.rejected += 1
        self.consec += 1
        loss = (info or {}).get('loss')
        ev = {'epoch': epoch, 'step_in_epoch': t, 'adam_step': step, 'consecutive': self.consec,
              'loss': loss if loss is None or not isinstance(loss, float) or math.isfinite(loss) else str(loss),
              'time': time.time()}
        self.events.append(ev)
        print('[UPDATE-REJECTED] epoch %d step %d adam_step %d consecutive %d loss %s' % (
            epoch, t, step, self.consec, loss), flush=True)
        if self.max_consec > 0 and self.consec >= self.max_consec:
            self._dump()
            raise TrainingFrozenError(
                '%d consecutive optimizer updates rejected (epoch %d, step %d, Adam step %d). Training would freeze. '
                'Run diag/probe_freeze.py on the last checkpoint to localise the first non-finite operation.'
                % (self.consec, epoch, t, step))
        return False

    def end_epoch(self, epoch):
        n = self.accepted + self.rejected
        rec = {'n_updates_accepted': self.accepted, 'n_updates_rejected': self.rejected,
               'adam_step': self.last_step, 'reject_frac': (self.rejected / n) if n else 0.0}
        self._dump(epoch)
        self.accepted = self.rejected = 0
        if self.max_frac < 1 and n and rec['reject_frac'] > self.max_frac:
            raise TrainingFrozenError('epoch %d: %.1f%% of optimizer updates rejected (> %.1f%%)' % (
                epoch, 100 * rec['reject_frac'], 100 * self.max_frac))
        return rec

    def _dump(self, epoch=None):
        if self.output_path:
            with open(os.path.join(self.output_path, 'rejected_updates.jsonl'), 'a') as f:
                for ev in self.events:
                    f.write(json.dumps(ev) + '\n')
            self.events = []
