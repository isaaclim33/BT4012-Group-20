// Exercise the actual template's storage adapter and decision handler without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/fraudrank/agents/dashboard_template.html'), 'utf8');
const storeCode = source.slice(source.indexOf('const store = {'), source.indexOf('const int = '));
const start = source.indexOf('    function save(goNext) {');
const saveCode = source.slice(start, source.indexOf('    decisionPanel.save = save;', start));

function context(cycle, fail, memory = new Map()) {
  const events = [];
  const ctx = {D: {meta: {storage_cycle: cycle}}, events,
    localStorage: {
      getItem: k => memory.get(k) ?? null,
      setItem: (k, v) => { if (fail) throw Error('quota'); memory.set(k, v); },
      removeItem: k => { if (fail) throw Error('disabled'); memory.delete(k); }
    },
    toast: text => events.push(text), chosen: () => 2, why: {value: 'Keep my reason', focus() {}},
    rec: 2, p: {id: 'P'}, next: {id: 'Q'}, STATUS: {2: 'Full audit opened'},
    logEvent() {}, go: id => events.push('go:' + id), route: () => events.push('route')};
  vm.createContext(ctx);
  vm.runInContext(storeCode + saveCode + '\nglobalThis.adapter = store; globalThis.saveDecision = save;', ctx);
  return ctx;
}

const failed = context('a', true);
failed.saveDecision(true);
assert.equal(failed.adapter.get('dec:P', null), null);
assert.equal(failed.why.value, 'Keep my reason');
assert.equal(failed.events.length, 1);
assert.match(failed.events[0], /Could not save/);
assert.equal(failed.adapter.remove('note:P'), false);

const memory = new Map();
const success = context('a', false, memory);
success.saveDecision(true);
assert.equal(success.adapter.get('dec:P', null).outcome, 2);
assert.equal(success.events.at(-1), 'go:Q');
assert.equal(context('b', false, memory).adapter.get('dec:P', null), null);
assert.equal(context('a', false, memory).adapter.get('dec:P', null).outcome, 2);
console.log('Dashboard storage: failure preserves input, success advances, cycles stay separate.');
