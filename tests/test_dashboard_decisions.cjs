// Exercise the template's actual decision filters, page, editing refresh and CSV.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/fraudrank/agents/dashboard_template.html'), 'utf8');
const start = source.indexOf('function recordedDecisions(');
const code = source.slice(start, source.indexOf('/* ---------- provider page ---------- */', start));
const saved = new Map();
const queue = [
  {id: 'P1', position: 1, lane: 'audit', action: 'priority_audit'},
  {id: 'P2', position: 2, lane: 'borderline', action: 'desk_review'},
  {id: 'P3', position: 3, lane: 'desk', action: 'desk_review'},
  {id: 'P4', position: 4, lane: 'audit', action: 'priority_audit'}
];
const app = {append(node) { this.content = node; }};
let clicked, blob;
function h(tag, attrs, ...children) {
  return {tag, attrs, children: children.flat(Infinity).filter(v => v != null && v !== false),
    click() { clicked = this; }, remove() {}};
}
const ctx = {QUEUE: queue, store: {get: (key, fallback) => saved.get(key) ?? fallback}, view: {decision: 0},
  STATUS: {2: 'Full audit opened', 3: 'Desk review', 4: 'No further action'},
  ACTION: {priority_audit: 'Full audit', desk_review: 'Desk review'}, h,
  hero() {}, $: () => app, icon: () => null, statusPill: n => ctx.STATUS[n],
  route: () => ctx.decisionsPage(), toast() {}, Blob, setTimeout() {},
  URL: {createObjectURL: value => { blob = value; return 'blob:test'; }, revokeObjectURL() {}},
  document: {body: {append() {}}}};
vm.createContext(ctx);
vm.runInContext(code, ctx);
const nodes = (node, predicate) => typeof node !== 'object' ? [] : [
  ...(predicate(node) ? [node] : []), ...node.children.flatMap(c => nodes(c, predicate))];
const text = node => typeof node !== 'object' ? String(node) : node.children.map(text).join(' ');
const rows = outcome => Array.from(ctx.recordedDecisions(outcome), ({p}) => p.id);

ctx.decisionsPage();
assert.match(text(app.content), /No decisions recorded yet/);
assert.equal(nodes(app.content, n => n.tag === 'button' && text(n) === 'Export saved decisions')[0].attrs.disabled, true);
saved.set('dec:P1', {outcome: 2, reason: 'Check records', t: 1000});
saved.set('dec:P2', {outcome: 2, reason: 'Escalate, with "records"\nattached', t: 3000});
saved.set('dec:P3', {outcome: 4, reason: '=2+3', t: 2000});
saved.set('st:P4', 1); // An in-progress case is not a recorded outcome.
assert.deepEqual(rows(0), ['P2', 'P3', 'P1']);
assert.deepEqual(rows(2), ['P2', 'P1']);
assert.deepEqual(rows(3), []);
ctx.decisionsPage();
assert.match(text(app.content), /Full audit opened \(2\)/);
assert.match(text(app.content), /Desk review \(0\)/);
assert.match(text(app.content), /Desk review \(borderline\)/);
assert.ok(nodes(app.content, n => n.tag === 'a' && n.attrs.href === '#/p/P2').length);
const auditFilter = nodes(app.content, n => n.tag === 'button' && text(n) === 'Full audit opened (2)')[0];
auditFilter.attrs.onclick();
assert.equal(ctx.view.decision, 2);
assert.equal(nodes(app.content, n => n.tag === 'tbody')[0].children.length, 2);
assert.match(text(app.content), /Showing 2 of 3 recorded decisions/);

// Editing an existing case changes its outcome list, not its original recommendation.
saved.set('dec:P2', {outcome: 3, reason: 'Desk review is sufficient', t: 4000});
ctx.decisionsPage();
assert.deepEqual(rows(2), ['P1']);
assert.deepEqual(rows(3), ['P2']);
assert.equal(queue[1].lane, 'borderline');
assert.match(text(app.content), /Full audit opened \(1\)/);
const csv = ctx.decisionsCsv(ctx.recordedDecisions());
assert.match(csv, /"Recorded At \(UTC\)"/);
assert.match(csv, /"'\=2\+3"/);
assert.ok(!csv.includes('P4'));
saved.get('dec:P2').reason = 'Escalate, with "records"\nattached';
assert.ok(ctx.decisionsCsv(ctx.recordedDecisions(3)).includes('"Escalate, with ""records""\nattached"'));
ctx.view.decision = 3;
ctx.exportDecisions();
assert.equal(clicked.attrs.download, 'case_decisions.csv');
blob.text().then(value => {
  assert.ok(value.includes('P2'));
  assert.ok(!value.includes('P1'));
  assert.ok(!value.includes('P3'));
  saved.clear();
  ctx.decisionsPage();
  assert.match(text(app.content), /No decisions recorded yet/);
  console.log('Decisions: filters, counts, empty state, edit refresh, links and filtered CSV passed.');
}).catch(error => { console.error(error); process.exitCode = 1; });
