// ICACT 2026 paper, IEEE two-column. Run: node paper/build_paper.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, SectionType,
  Table, TableRow, TableCell, WidthType, ShadingType, ImageRun, convertInchesToTwip,
} = require("docx");

const F = "Times New Roman";
const FIGS = "paper/figs";
const EXP = "experiments/final_run_2026-08-11/figs";

// IEEE sizes are in half-points: 10pt = 20
const body = (t, opts = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: 220 },
  indent: { firstLine: convertInchesToTwip(0.2) },
  children: [new TextRun({ text: t, font: F, size: 20, ...opts })],
});
const sec = (n, t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: `${n}. ${t}`, font: F, size: 20, allCaps: true })],
});
const sub = (t) => new Paragraph({
  spacing: { before: 120, after: 60 },
  children: [new TextRun({ text: t, font: F, size: 20, italics: true })],
});
const fig = (path, w, h, caption) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path),
      transformation: { width: w, height: h } })] }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 140 },
    children: [new TextRun({ text: caption, font: F, size: 16 })] }),
];
function tbl(caption, headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, bold, fill) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: String(text), font: F, size: 15, bold })] })] });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 60 },
      children: [new TextRun({ text: caption, font: F, size: 16 })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ children: headers.map((h, i) => cell(h, widths[i], true, "E8E8E8")) }),
        ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }))] }),
    new Paragraph({ spacing: { after: 140 }, children: [new TextRun({ text: " ", size: 8 })] }),
  ];
}

// ---------- Title block (single column) ----------
const head = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Full Carbon Accounting Changes the Conclusion: Measured-Energy Evaluation of Confidence-Aware LLM Cascades", font: F, size: 48 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
    children: [new TextRun({ text: "Tharusha Rasath Hemachandra", font: F, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
    children: [new TextRun({ text: "Faculty of Computing, NSBM Green University, Homagama, Sri Lanka", font: F, size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "tharusharasathml@gmail.com", font: F, size: 20 })] }),
];

// ---------- Two-column body ----------
const col = [];

col.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 },
  children: [
    new TextRun({ text: "Abstract—", font: F, size: 18, bold: true, italics: true }),
    new TextRun({ text: "Model cascading reduces the cost of large language model inference by answering easy queries with a small model and escalating only uncertain ones. Published evaluations of such systems charge each query only for the model that produced the returned answer, silently discarding the energy already spent on the small-model attempt that triggered escalation. This paper quantifies the consequence of that convention using physically measured energy rather than estimates. We implement GreenGate, an open-source cascading middleware that meters GPU power at 100 ms across all devices, applies temperature-scaled uncertainty as a training-free routing signal, and charges escalated queries for both tiers. Across 2,800 evaluated queries spanning open-ended user traffic, a structured benchmark, and a visual question-answering workload, and across three deployment configurations, we find that the prevailing convention overstates carbon savings by 9.7 percent at low escalation and reports a saving of 54.2 percent where full accounting shows a 15.2 percent increase. Full accounting yields a break-even condition in which savings approximate the small-tier cost complement minus the escalation rate, which our measurements obey. Under this accounting the cascade still meets a 30 percent carbon reduction at 95 percent quality retention on both text workloads. A five-model study further shows that routing-signal informativeness scales with small-model capability, and identifies a capability floor below which cascading is strictly harmful, a regime the prevailing convention cannot detect.", font: F, size: 18, italics: true }),
  ] }));
col.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
  children: [
    new TextRun({ text: "Keywords—", font: F, size: 18, bold: true, italics: true }),
    new TextRun({ text: "green AI, model cascading, carbon accounting, energy measurement, uncertainty estimation, sustainable computing", font: F, size: 18, italics: true }),
  ] }));

// I. Introduction
col.push(sec("I", "Introduction"));
col.push(body("Inference, not training, now dominates the lifecycle energy of deployed language models, and the gap widens as adoption grows [1], [2]. A structural inefficiency underlies this cost: production systems route every request to the same large model regardless of difficulty, so a trivial factual query consumes the same per-token energy as one requiring extended reasoning. Cascading addresses this by attempting a small model first and escalating only when some confidence signal indicates the answer is unreliable [3], [4].", { }));
col.push(body("Reported savings for such systems are substantial, but they rest on an accounting convention that has not been examined. When a query escalates, the small model has already run: its energy is spent, its answer discarded. Prevailing evaluations charge the query only for the large model that produced the returned answer. The discarded computation is invisible in the reported total."));
col.push(body("This paper asks what happens when it is not. We make three contributions. First, we implement full carbon accounting, charging every escalated query for both tiers, and quantify the divergence from the prevailing convention using measured rather than estimated energy. Second, we derive and empirically confirm a break-even condition that determines when cascading reduces emissions at all. Third, we characterise how the routing signal itself behaves across task structure and across five small models, establishing a capability floor below which cascading increases emissions. All code, raw per-query records and analysis scripts are public, and the system is installable as a Python package."));

// II. Related work
col.push(sec("II", "Related Work"));
col.push(body("FrugalGPT [3] established the cascade pattern for commercial APIs, optimising monetary cost with a learned scoring function. AutoMix [4] replaced the external scorer with few-shot self-verification and modelled routing as a partially observable decision process. C3PO [5] provided label-free cost guarantees using conformal prediction. All three optimise cost or accuracy rather than energy, and none report the energy of discarded small-model runs."));
col.push(body("Energy-aware routing is more recent. GreenServ [6] trains contextual bandits for energy-accuracy trade-offs across a model pool, and GAR [7] formulates carbon-aware routing under accuracy and latency constraints. Both estimate energy from analytical models rather than measuring it, both require training on a fixed model pool, and neither is publicly installable. EcoLogits [8] supplies peer-reviewed per-token carbon estimation for API models but performs no routing."));
col.push(body("On the signal side, temperature scaling [9] remains the standard correction for neural overconfidence, and semantic entropy [10] improves uncertainty estimation by clustering sampled generations by meaning at the cost of repeated inference. Our work is distinguished by measuring energy on hardware, by accounting for discarded computation, and by extending confidence routing to a visual workload."));

// III. Method
col.push(sec("III", "Method"));
col.push(sub("A. Cascade and routing signal"));
col.push(body("A query is first answered by a small open-weight model executed locally, which is required because the routing signal is derived from token logits that hosted APIs do not expose. For free-form text we compute mean token entropy over the generated sequence; for multiple-choice items we compute entropy over the renormalised distribution across answer options; for visual question answering we use the mean probability the model assigned to its own generated tokens. Logits are divided by a temperature fitted on a held-out validation split before the softmax [9]. The query escalates when the resulting signal exceeds a threshold and a carbon budget permits."));
col.push(sub("B. Dual-mode carbon measurement"));
col.push(body("Local inference is measured, not estimated. A background thread polls instantaneous power draw at 100 ms intervals through NVML, summed across every visible device, since quantised models sharded over multiple accelerators draw power on all of them; sampling a single device understated an identical workload by a factor of 2.4 in our own instrumentation audit. Energy is mean sampled power integrated over inference time, and carbon follows C = E x PUE x I with PUE 1.2 and grid intensity I. API tiers, which cannot be metered, use EcoLogits [8] and are reported separately as estimates."));
col.push(sub("C. Full carbon accounting and break-even"));
col.push(body("Let C_s and C_l denote the mean per-query carbon of the small and large tiers and let e be the escalation rate. The prevailing convention reports C_naive = (1-e) C_s + e C_l. Full accounting charges the discarded attempt on every escalated query, giving C_full = C_s + e C_l. Relative to always using the large tier, savings under full accounting are"));
col.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  children: [new TextRun({ text: "S = 1 - (C_s / C_l) - e,", font: F, size: 20, italics: true })] }));
col.push(body("so a cascade reduces emissions only while e < 1 - C_s/C_l. The inter-tier energy ratio therefore bounds achievable savings, and a sufficiently escalating cascade emits more than never cascading. The naive convention omits the C_s term on escalated queries and cannot express this condition.", { }));

// IV. Setup
col.push(sec("IV", "Experimental Setup"));
col.push(body("We evaluate three deployment configurations on shared query sets. The primary local-to-local configuration pairs Qwen2.5-1.5B-Instruct with Mistral-7B-Instruct-v0.2 in 4-bit, both metered on identical hardware, which is the only configuration in which both tiers are measured on a common basis. A local-to-API configuration pairs Mistral-7B with GPT-4o-mini, exercising both profiler modes. An API-to-API configuration pairs GPT-4o-mini with GPT-4o, routed on provider log-probabilities, where the exactly known quantity is monetary cost."));
col.push(body("Workloads are 500 first-turn ShareGPT user queries scrubbed of personal information, 300 MMLU test questions, and 500 VQAv2 questions with embedded images; 150 held-out MMLU validation questions per model are used only for calibration. Open-ended answers are scored once each on a five-point scale by a GPT-4o judge, cross-checked by an open-weight second judge and by a blind two-annotator human sample. Structured and visual workloads use exact-match ground truth. All experiments ran on free-tier dual NVIDIA T4 accelerators; energy repeatability across three runs of a fixed workload was 1.4 percent."));
col.push(body("Every model answers every query once; routing policies, threshold sweeps, grid conditions and ablations are then simulated offline from cached per-query records, so all policies are compared on identical data. Baselines are always-large, always-small, random escalation, and a blind cascade escalating at random at the same rate as the entropy gate, which isolates the signal's contribution."));

// V. Results
col.push(sec("V", "Results"));
col.push(sub("A. The accounting convention changes conclusions"));
col.push(body("At the operating point maximising quality per gram on ShareGPT, full accounting reports a 47.4 percent carbon reduction where the naive convention reports 54.2 percent, an overstatement of 9.7 percent in relative terms. The divergence grows with escalation. Fig. 1 shows both curves against the always-large baseline: the naive curve lies systematically to the left of the truth, and beyond roughly 75 percent escalation the full-accounting curve crosses to the right of the baseline, meaning the cascade emits more than never cascading while the naive convention still reports a saving of 15.2 percent."));
col.push(...fig(`${FIGS}/fig_accounting_pareto.png`, 240, 184,
  "Fig. 1. Quality-carbon frontier on ShareGPT under both accounting conventions. Naive accounting understates carbon at every operating point, and at high escalation reports savings where full accounting shows an increase over always-large."));
col.push(sub("B. Savings under honest accounting"));
col.push(body("Under full accounting the cascade still meets a demanding target on both text workloads. On ShareGPT it reduces carbon by 37.5 percent while retaining 95.9 percent of the large tier's judged quality, escalating 15 percent of queries; relaxing retention to 94.8 percent yields 47.6 percent reduction at 5 percent escalation, close to the 52.8 percent ceiling implied by the measured 2.12x energy ratio. On MMLU it reduces carbon by 29.5 percent at 95.0 percent retention. Mean latency falls by 39 percent, since most queries never reach the larger model. Table I summarises the primary configuration."));
col.push(...tbl("TABLE I. SHAREGPT, 500 QUERIES, FULL ACCOUNTING",
  ["Policy", "Qual.", "Ret.", "CO2 cut", "Esc."],
  [["Always-large (Mistral-7B)", "3.056", "100%", "-", "100%"],
   ["Always-small (Qwen-1.5B)", "2.874", "94.0%", "52.8%", "0%"],
   ["Random 50%", "2.932", "95.9%", "26.2%", "50%"],
   ["Blind cascade", "2.864", "93.7%", "48.6%", "4%"],
   ["GreenGate (t=2.09)", "2.930", "95.9%", "37.5%", "15%"],
   ["GreenGate (t=2.55)", "2.894", "94.7%", "47.4%", "5%"]],
  [1500, 620, 620, 780, 620]));
col.push(sub("C. Signal validity depends on task structure"));
col.push(body("The routing signal is informative where answers are constrained and weak where they are not. On MMLU, questions the small model answers incorrectly carry 0.331 bits more choice entropy than those it answers correctly, and on VQAv2 mean confidence is 0.940 when correct against 0.846 when wrong. On open-ended generation, by contrast, the signal is near chance at identifying poor answers: tie-corrected AUROC is 0.539 for mean token entropy and 0.439 for semantic entropy with three samples at 4.3x the energy, and 0.536 for API log-probabilities. Answers where all three samples agreed were still judged poor half the time, indicating consistent rather than detectable error. On such workloads the cascade's savings derive from its structure rather than from selective routing, which the blind-cascade baseline confirms."));
col.push(sub("D. Signal informativeness scales with capability"));
col.push(body("Evaluating five small models spanning four families against a fixed large tier shows that the signal is a property of the model computing it rather than of entropy as such. Fig. 2 plots the entropy gap against small-model accuracy: it rises from 0.017 bits for TinyLlama-1.1B to 0.500 bits for Qwen2.5-3B. TinyLlama scores 24.3 percent on four-choice MMLU, below chance, and its cascade emits 9.5 percent more carbon than never cascading, a regime naive accounting cannot surface because it never charges the discarded attempt. Two small models, Phi-3.5-mini and Qwen2.5-3B, outperform the 7B large tier outright, giving both higher accuracy and 44 to 53 percent lower carbon and showing that tier ordering follows model quality rather than parameter count."));
col.push(...fig(`${FIGS}/fig_signal_vs_capability.png`, 240, 176,
  "Fig. 2. Routing-signal informativeness against small-model capability on MMLU. Below the chance line the signal carries no information and cascading increases emissions."));
col.push(sub("E. Measurement basis and deployment"));
col.push(body("The local-to-API configuration measured 78.8 g for the local tier against an estimated 3.5 g for the API tier over the same queries. These figures are not commensurable: one is a hardware measurement of a dedicated 2018-generation accelerator, the other a model-based estimate amortising modern batched infrastructure. The practical implication is that whether local-first routing reduces carbon is a property of deployment hardware relative to the provider's, which a dual-mode profiler makes decidable per deployment rather than assumable. In the API-to-API configuration, where carbon is estimated on both sides, the exactly known result is financial: the cascade retains 98.0 percent of quality at 77.0 percent lower cost."));

// VI. Discussion
col.push(sec("VI", "Discussion and Limitations"));
col.push(body("Our results support a narrower claim than the cascade literature typically makes. Cascading is not uniformly green; it is green under a stated condition that practitioners can now measure. Reporting that condition requires charging for discarded computation, and doing so changes reported outcomes materially, including reversing the sign of the conclusion in an identifiable regime."));
col.push(body("Several limitations bound these findings. All local measurements come from one hardware configuration, so absolute figures, though not the break-even relationship, are hardware-specific. The large tier is a 7B model rather than a frontier-scale one, which caps the inter-tier energy ratio and hence achievable savings. Judged quality on open-ended text is a model's assessment on an ordinal scale; our two-annotator human sample showed low exact agreement driven by severity differences, though the retention ratio, which is what we report, was stable within 5.5 percentage points across both annotators and the judge. Semantic entropy was evaluated at three samples for cost reasons, which limits its resolution."));

// VII. Conclusion
col.push(sec("VII", "Conclusion"));
col.push(body("We showed that the accounting convention used to evaluate model cascades materially overstates their carbon savings, and that correcting it exposes a break-even condition beyond which cascading increases emissions. Under honest accounting and physically measured energy, confidence-aware cascading still delivers 37.5 percent carbon reduction at 95.9 percent quality retention on real user traffic, and reduces escalation to 5 percent on a visual workload while exceeding both static tiers' accuracy. We also showed that the routing signal is informative only above a small-model capability threshold, below which cascading is strictly harmful. The system, its raw per-query records and its analysis scripts are publicly available, and the library installs from the Python Package Index."));

// References
col.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: "References", font: F, size: 20, allCaps: true })] }));
const refs = [
  "A. S. Luccioni, S. Viguier, and A.-L. Ligozat, \"Power hungry processing: Watts driving the cost of AI deployment?\" in Proc. ACM FAccT, 2024.",
  "A. A. Chien, L. Lin, H. Nguyen, V. Rao, T. Sharma, and R. Wijayawardana, \"Reducing the carbon impact of generative AI inference (today and in 2035),\" in Proc. 2nd Workshop on Sustainable Computer Systems (HotCarbon), ACM, 2023.",
  "L. Chen, M. Zaharia, and J. Zou, \"FrugalGPT: How to use large language models while reducing cost and improving performance,\" arXiv:2305.05176, 2023.",
  "P. Aggarwal, A. Madaan, A. Anand, et al., \"AutoMix: Automatically mixing language models,\" in Proc. NeurIPS, 2024.",
  "S. Chen et al., \"C3PO: Label-free cascade with conformal cost guarantees,\" 2025.",
  "A. Ziller et al., \"GreenServ: Energy-efficient context-aware dynamic routing for multi-model LLM inference,\" in Proc. ICPE, 2026.",
  "A. Barrak et al., \"GAR: Carbon-aware routing for LLM inference with service level objectives,\" 2025.",
  "O. Rince et al., \"EcoLogits: Tracking the energy and carbon footprint of generative AI APIs,\" Journal of Open Source Software, 2025.",
  "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On calibration of modern neural networks,\" in Proc. ICML, 2017.",
  "S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, \"Detecting hallucinations in large language models using semantic entropy,\" Nature, vol. 630, 2024.",
];
refs.forEach((r, i) => col.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 20 },
  indent: { left: convertInchesToTwip(0.22), hanging: convertInchesToTwip(0.22) },
  children: [new TextRun({ text: `[${i + 1}] ${r}`, font: F, size: 16 })] })));

const doc = new Document({
  styles: { default: { document: { run: { font: F, size: 20 } } } },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 },
        margin: { top: 1000, bottom: 1000, left: 1000, right: 1000 } } }, children: head },
    { properties: { type: SectionType.CONTINUOUS,
        column: { count: 2, space: 340, equalWidth: true },
        page: { size: { width: 11906, height: 16838 },
          margin: { top: 1000, bottom: 1000, left: 1000, right: 1000 } } }, children: col },
  ],
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("paper/ICACT2026_GreenGate.docx", b);
  console.log("wrote paper/ICACT2026_GreenGate.docx");
});
