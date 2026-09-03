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
      children: [new TextRun({ text: String(text), font: F, size: 16, bold })] })] });
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
    children: [new TextRun({ text: "What Counts as Saved? Measured-Energy Accounting for Confidence-Aware LLM Cascades", font: F, size: 48 })] }),
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
    new TextRun({ text: "Model cascading reduces the cost of large language model inference by answering easy queries with a small model and escalating only uncertain ones. When a query escalates, the small model has already run and its energy is spent, yet an escalated query can be charged either for both tiers or only for the tier that produced the returned answer. This paper measures what that choice is worth. We implement GreenGate, an open-source cascading middleware that meters GPU power at 100 ms intervals across all devices, uses temperature-scaled uncertainty as a training-free routing signal, and charges escalated queries for both tiers. Across 2,800 evaluated queries spanning open-ended user traffic, a structured benchmark and a visual question-answering workload, the two conventions diverge by 2.6 percentage points at 5 percent escalation and by 31.4 points at 60 percent, where charging only the answering tier reports a 23.3 percent reduction while full accounting shows an 8.0 percent increase over never cascading. Full accounting yields a break-even condition in which savings approximate the complement of the inter-tier energy ratio minus the escalation rate, which our measurements obey. Under this stricter accounting the cascade still reduces measured energy by 37.3 percent (95 percent CI 33.5 to 40.9) while retaining 95.9 percent of large-tier answer quality. A five-model study shows that routing-signal informativeness scales with small-model capability and identifies a capability floor below which cascading is strictly harmful, a regime that charging only the answering tier cannot express.", font: F, size: 18, italics: true }),
  ] }));
col.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
  children: [
    new TextRun({ text: "Keywords—", font: F, size: 18, bold: true, italics: true }),
    new TextRun({ text: "green AI, model cascading, carbon accounting, energy measurement, uncertainty estimation", font: F, size: 18, italics: true }),
  ] }));

// I. Introduction
col.push(sec("I", "Introduction"));
col.push(body("Inference, not training, now dominates the lifecycle energy of deployed language models, and the gap widens as adoption grows [1], [2]. A structural inefficiency underlies this cost: production systems route every request to the same large model regardless of difficulty, so a trivial factual query consumes the same per-token energy as one requiring extended reasoning. Cascading addresses this by attempting a small model first and escalating only when some confidence signal indicates the answer is unreliable [3], [4].", { }));
col.push(body("The idea is simple, and so is the question this paper asks. A small model tries first. If it is uncertain, a large model answers instead. But the small model has already run, and its energy is already spent. An evaluation can charge that escalated query for both models, or only for the one that produced the answer it returned. Fig. 1 illustrates the difference. Both conventions are defensible ways of describing the same system; they are not equivalent, and how far apart they are has not been measured."));
col.push(body("We measure it. Because the difference is a quantity of energy, answering the question requires metering hardware rather than applying published per-token estimates, and it requires evaluating the same routing decisions under both conventions on identical data."));
col.push(body("We make three contributions. First, we quantify the divergence between the two conventions using physically measured energy, and show it is small at low escalation but large enough at moderate escalation to reverse the sign of the reported conclusion. Second, we derive and empirically confirm a break-even condition that determines when cascading reduces energy at all. Third, we characterise the routing signal across task structure and across five small models, establishing a capability floor below which cascading is strictly harmful. All code, raw per-query records and analysis scripts are public, and the system installs from the Python Package Index."));
col.push(...fig(`${FIGS}/fig_accounting_concept.png`, 240, 176,
  "Fig. 1. The two accounting conventions for an escalated query. Both describe the same execution; they differ in whether the already-spent small-model energy appears in the reported total."));

// II. Related work
col.push(sec("II", "Related Work"));
col.push(body("FrugalGPT [3] established the cascade pattern for commercial APIs, optimising monetary cost with a learned scoring function. AutoMix [4] replaced the external scorer with few-shot self-verification and modelled routing as a partially observable decision process. C3PO [5] provided label-free cost guarantees using conformal prediction. These systems optimise monetary cost or accuracy rather than energy, so the question this paper studies does not arise for them in the same form: where cost is billed per API call, expenditure on an escalated query's first attempt is incurred whether or not it is separately reported. We therefore make no claim about how any individual prior system accounts internally, and note only that published descriptions do not report the energy attributable to superseded attempts, which is the quantity we measure."));
col.push(body("Energy-aware routing is more recent. GreenServ [6] trains contextual bandits for energy-accuracy trade-offs across a model pool, and GAR [7] formulates carbon-aware routing under accuracy and latency constraints. Both estimate energy from analytical models rather than measuring it, both require training on a fixed model pool, and neither is publicly installable. EcoLogits [8] supplies peer-reviewed per-token carbon estimation for API models but performs no routing."));
col.push(body("On the signal side, temperature scaling [9] remains the standard correction for neural overconfidence, and semantic entropy [10] improves uncertainty estimation by clustering sampled generations by meaning at the cost of repeated inference. Table I positions this work against the closest systems along the axes relevant here, as described in their published accounts. The combination distinguishing this study is that energy is measured on hardware rather than estimated, that superseded attempts are explicitly charged and the alternative convention reported alongside, and that confidence routing is extended to a visual workload."));
col.push(...tbl("TABLE I. POSITIONING, AS DESCRIBED IN PUBLISHED ACCOUNTS",
  ["System", "Optimises", "Energy figures", "Training-free", "Superseded attempt reported"],
  [["FrugalGPT [3]", "Monetary cost", "None", "No", "Not reported"],
   ["AutoMix [4]", "Cost, accuracy", "None", "Yes", "Not reported"],
   ["C3PO [5]", "Cost bound", "None", "Yes", "Not reported"],
   ["GreenServ [6]", "Energy", "Estimated", "No", "Not reported"],
   ["GAR [7]", "Carbon, SLO", "Estimated", "No", "Not reported"],
   ["EcoLogits [8]", "n/a (estimator)", "Estimated", "Yes", "n/a"],
   ["This work", "Energy, quality", "Measured", "Yes", "Reported, both conventions"]],
  [1140, 900, 820, 740, 1160]));

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
col.push(body("so a cascade reduces energy only while e < 1 - C_s/C_l. The inter-tier energy ratio therefore bounds achievable savings, and a sufficiently escalating cascade emits more than never cascading. The naive convention omits the C_s term on escalated queries and cannot express this condition.", { }));
col.push(sub("D. Calibration and threshold selection"));
col.push(body("Language models are systematically overconfident, so raw entropy thresholds are not comparable across models. We fit a single scalar temperature per small model by minimising negative log-likelihood on a held-out validation split, following [9], and report Expected Calibration Error before and after. Across the seven models calibrated in this study, ECE fell by between 32.9 and 71.9 percent; the 4-bit large model required the largest correction, consistent with quantisation amplifying overconfidence. Thresholds are then selected as percentiles of the observed signal distribution, which makes them transferable across models whose entropy scales differ."));
col.push(...fig(`${FIGS}/fig_calibration.png`, 240, 170,
  "Fig. 2. Expected Calibration Error before and after temperature scaling for the seven models calibrated in this study. Every model is overconfident before correction; the 4-bit large model requires the largest temperature."));
col.push(sub("E. Carbon budget"));
col.push(body("Deployments that must respect a carbon ceiling cannot rely on a static threshold, since escalation rate varies with traffic. We maintain a sliding window over recent routing decisions and block escalation when the projected cost would exceed the budget within the window, allowing it to resume as older spending ages out. Replaying our judged queries under arrival timestamps drawn from a production inference trace, a 10 g per 600 s budget deferred 37 of 51 attempted escalations during bursts, capping carbon 15 percent below the unbudgeted operating point while mean quality degraded from 2.464 to 2.390. The mechanism therefore trades quality gracefully rather than failing, which is the behaviour a service-level constraint requires."));

// IV. Setup
col.push(sec("IV", "Experimental Setup"));
col.push(body("We evaluate three deployment configurations on shared query sets. The primary local-to-local configuration pairs Qwen2.5-1.5B-Instruct with Mistral-7B-Instruct-v0.2 in 4-bit, both metered on identical hardware, which is the only configuration in which both tiers are measured on a common basis. A local-to-API configuration pairs Mistral-7B with GPT-4o-mini, exercising both profiler modes. An API-to-API configuration pairs GPT-4o-mini with GPT-4o, routed on provider log-probabilities, where the exactly known quantity is monetary cost."));
col.push(body("Workloads are 500 first-turn ShareGPT user queries scrubbed of personal information, 300 MMLU test questions, and 500 VQAv2 questions with embedded images; 150 held-out MMLU validation questions per model are used only for calibration. Open-ended answers are scored once each on a five-point scale by a GPT-4o judge, cross-checked by an open-weight second judge and by a blind two-annotator human sample. Structured and visual workloads use exact-match ground truth. All experiments ran on free-tier dual NVIDIA T4 accelerators; energy repeatability across three runs of a fixed workload was 1.4 percent."));
col.push(body("Every model answers every query once; routing policies, threshold sweeps, grid conditions and ablations are then simulated offline from cached per-query records, so all policies are compared on identical data and the nineteen-point threshold sweep costs no additional inference. Baselines are always-large, always-small, random escalation, and a blind cascade escalating at random at the same rate as the entropy gate, which isolates the signal's contribution from the cascade structure itself."));
col.push(body("Reproducibility was treated as a design requirement rather than an afterthought. Seeds are fixed, every intermediate record is written to disk as it is produced, and each reported number is regenerable from those records by a published script. During development an audit of our own instrumentation against its written specification revealed two defects, single-device sampling and two-point rather than continuous polling, whose correction changed measured energy for an identical workload by a factor of 2.4 and required re-collecting all measured data; a third defect in an analysis routine inflated one ablation statistic before being caught by cross-checking against its own plot. We report these because they bear directly on the reliability of energy figures in this literature, most of which are estimated and therefore not subject to the same failure modes or the same checks."));

// V. Results
col.push(sec("V", "Results"));
col.push(sub("A. The accounting convention changes conclusions"));
col.push(body("The two conventions agree closely when escalation is rare and diverge sharply when it is not, as Table II shows for the primary configuration. At 5 percent escalation the gap is 2.6 percentage points, a 5.5 percent relative overstatement. At 60 percent escalation the gap reaches 31.4 points and the conventions disagree in sign: charging only the answering tier reports a 23.3 percent energy reduction, while charging both reports an 8.0 percent increase over never cascading at all. A bootstrap over queries (2,000 resamples) puts the gap at the deployed operating point at 2.7 points with a 95 percent confidence interval of 1.7 to 3.8, so the divergence is small but not attributable to sampling. Fig. 3 shows the effect on the full frontier: the curve computed under the answering-tier convention lies systematically left of the measured truth, and at high escalation the full-accounting curve crosses to the right of the always-large baseline."));
col.push(...tbl("TABLE II. DIVERGENCE BETWEEN CONVENTIONS, SHAREGPT",
  ["Escalation", "Full accounting", "Answering tier only", "Gap (pp)"],
  [["5%", "47.6%", "50.2%", "2.6"],
   ["10%", "42.5%", "47.6%", "5.1"],
   ["20%", "32.5%", "42.9%", "10.3"],
   ["40%", "12.3%", "33.5%", "21.2"],
   ["60%", "-8.0%", "23.3%", "31.4"],
   ["80%", "-27.8%", "12.4%", "40.2"]],
  [1000, 1180, 1300, 800]));
col.push(...fig(`${FIGS}/fig_accounting_pareto.png`, 240, 184,
  "Fig. 3. Quality-carbon frontier on ShareGPT under both accounting conventions. Naive accounting understates carbon at every operating point, and at high escalation reports savings where full accounting shows an increase over always-large."));
col.push(sub("B. Savings under honest accounting"));
col.push(body("Under full accounting the cascade still meets a demanding target on both text workloads. On ShareGPT it reduces measured energy by 37.3 percent (95% CI 33.5-40.9) while retaining 95.9 percent of the large tier's judged quality, escalating 15 percent of queries; relaxing retention to 94.8 percent yields 47.4 percent reduction (95% CI 44.6-49.9) at 5 percent escalation, close to the 52.8 percent ceiling implied by the measured 2.12x energy ratio. On MMLU it reduces measured energy by 29.5 percent at 95.0 percent retention. Mean latency falls by 39 percent, since most queries never reach the larger model. Table III summarises the primary configuration."));
col.push(...tbl("TABLE III. SHAREGPT, 500 QUERIES, FULL ACCOUNTING",
  ["Policy", "Qual.", "Ret.", "CO2 cut", "Esc."],
  [["Always-large (Mistral-7B)", "3.056", "100%", "-", "100%"],
   ["Always-small (Qwen-1.5B)", "2.874", "94.0%", "52.8%", "0%"],
   ["Random 50%", "2.932", "95.9%", "26.2%", "50%"],
   ["Blind cascade", "2.864", "93.7%", "48.6%", "4%"],
   ["GreenGate (t=2.09)", "2.930", "95.9%", "37.5%", "15%"],
   ["GreenGate (t=2.55)", "2.894", "94.7%", "47.4%", "5%"]],
  [1500, 620, 620, 780, 620]));
col.push(body("The structured workload behaves similarly under the same accounting. Table IV reports MMLU, where the measured inter-tier energy ratio of 4.73x admits a higher ceiling; the entropy gate reaches 29.5 percent reduction at 95.0 percent retention, and an oracle router that escalates exactly when the small model is wrong and the large model right would reach 70.3 percent accuracy, above the large tier itself, at 55.0 percent lower carbon. That gap between the deployed gate and the oracle quantifies the headroom remaining for better routing signals."));
col.push(...tbl("TABLE IV. MMLU, 300 QUESTIONS, FULL ACCOUNTING",
  ["Policy", "Acc.", "Ret.", "CO2 cut", "Esc."],
  [["Always-large (Mistral-7B)", "0.533", "100%", "-", "100%"],
   ["Always-small (Qwen-0.5B)", "0.470", "88.1%", "78.9%", "0%"],
   ["Random 50%", "0.497", "93.1%", "38.9%", "50%"],
   ["Blind cascade", "0.500", "93.8%", "19.4%", "62%"],
   ["GreenGate (calibrated)", "0.507", "95.0%", "29.5%", "45%"],
   ["Oracle (upper bound)", "0.703", "131.9%", "55.0%", "23%"]],
  [1500, 620, 620, 780, 620]));
col.push(sub("C. Signal validity depends on task structure"));
col.push(body("The routing signal is informative where answers are constrained and weak where they are not. On MMLU, questions the small model answers incorrectly carry 0.331 bits more choice entropy than those it answers correctly, and on VQAv2 mean confidence is 0.940 when correct against 0.846 when wrong. On open-ended generation, by contrast, the signal is near chance at identifying poor answers: tie-corrected AUROC is 0.539 for mean token entropy and 0.439 for semantic entropy with three samples at 4.3x the energy, and 0.536 for API log-probabilities. Answers where all three samples agreed were still judged poor half the time, indicating consistent rather than detectable error. On such workloads the cascade's savings derive from its structure rather than from selective routing, which the blind-cascade baseline confirms."));
col.push(sub("D. Signal informativeness scales with capability"));
col.push(body("Evaluating five small models spanning four families against a fixed large tier shows that the signal is a property of the model computing it rather than of entropy as such. Fig. 4 plots the entropy gap against small-model accuracy: it rises from 0.017 bits for TinyLlama-1.1B to 0.500 bits for Qwen2.5-3B. TinyLlama scores 24.3 percent on four-choice MMLU, below chance, and its cascade consumes 9.5 percent more energy than never cascading, a regime naive accounting cannot surface because it never charges the discarded attempt. Two small models, Phi-3.5-mini and Qwen2.5-3B, outperform the 7B large tier outright, giving both higher accuracy and 44 to 53 percent lower carbon and showing that tier ordering follows model quality rather than parameter count."));
col.push(...fig(`${FIGS}/fig_signal_vs_capability.png`, 240, 176,
  "Fig. 4. Routing-signal informativeness against small-model capability on MMLU. Below the chance line the signal carries no information and cascading increases emissions."));
col.push(...tbl("TABLE V. FIVE SMALL MODELS AGAINST A FIXED 7B LARGE TIER",
  ["Small model", "Acc.", "Gap (bits)", "Ratio", "Cut at 95% ret."],
  [["Phi-3.5-mini (3.8B)", "0.667", "0.440", "1.95x", "44.2%"],
   ["Qwen2.5-3B", "0.653", "0.500", "2.42x", "52.6%"],
   ["Qwen2.5-0.5B", "0.470", "0.331", "4.73x", "33.6%"],
   ["SmolLM2-1.7B", "0.423", "0.348", "4.66x", "5.9%"],
   ["TinyLlama-1.1B", "0.243", "0.017", "4.57x", "-9.5%"]],
  [1420, 620, 800, 660, 940]));
col.push(sub("E. Vision routing"));
col.push(body("On 500 VQAv2 questions the confidence signal separates outcomes clearly: mean token probability is 0.940 when the small vision model answers correctly and 0.846 when it does not. Routing on this signal escalates only the least confident 5 percent of queries and reaches 67.8 percent accuracy, exceeding both the local model alone at 67.0 percent and the commercial API tier at 52.4 percent. The small model's advantage partly reflects genuine specialisation on this task format and partly a sensitivity of exact-match scoring to the API model's more verbose phrasing, which we note rather than adjust for. The result nonetheless demonstrates that confidence routing transfers to a modality where per-query energy is substantially higher, and that a well-chosen small tier can dominate a larger one on both accuracy and carbon simultaneously."));
col.push(sub("F. Reliability of quality measurement"));
col.push(body("Because open-ended quality is scored by a model, we applied two checks. An open-weight second judge re-scored 200 answers and agreed with the primary judge within one point in 70 percent of cases, with the primary judge systematically stricter, and stricter on small-tier answers than large-tier ones, which biases against the cascade rather than for it. A blind human sample of 100 answers scored by two annotators produced low exact agreement, driven by a severity difference between annotators, but the retention ratio we actually report was stable across raters at 88.4, 82.9 and 86.8 percent for the two annotators and the judge respectively. Since retention is a ratio of tier means, systematic severity cancels, and the reported figures are robust to which rater is used even though absolute scores are not."));
col.push(sub("G. Grid intensity and measurement basis"));
col.push(body("Because both tiers of the primary configuration are measured in joules, carbon under any grid intensity is a linear rescaling: the 47.4 percent relative reduction is invariant across green (50 gCO2/kWh), world-average (475) and coal-heavy (800) conditions, while absolute savings range from 4.4 g to 69.9 g per 500 queries. Grid intensity therefore sets the magnitude, not the merit, of cascading in homogeneous deployments."));
col.push(body("Mixed deployments are different. The local-to-API configuration measured 78.8 g for the local tier against an estimated 3.5 g for the API tier over the same queries. These figures are not commensurable: one is a hardware measurement of a dedicated 2018-generation accelerator, the other a model-based estimate amortising modern batched infrastructure. The practical implication is that whether local-first routing reduces carbon is a property of deployment hardware relative to the provider's, which a dual-mode profiler makes decidable per deployment rather than assumable. In the API-to-API configuration, where carbon is estimated on both sides, the exactly known result is financial: the cascade retains 98.0 percent of quality at 77.0 percent lower cost."));

// VI. Discussion
col.push(sec("VI", "Discussion and Limitations"));
col.push(body("Our results support a narrower claim than the cascade literature typically makes. Cascading is not uniformly green; it is green under a stated condition that practitioners can now measure. Reporting that condition requires charging for discarded computation, and doing so changes reported outcomes materially, including reversing the sign of the conclusion in an identifiable regime."));
col.push(body("The practical guidance that follows is threefold. Practitioners should measure their inter-tier energy ratio before adopting a cascade, since it bounds attainable savings; should monitor escalation rate against the break-even condition rather than assuming a fixed threshold remains beneficial as traffic shifts; and should verify that their small tier clears a capability threshold on their own workload, because below it the cascade is a pure loss. Our implementation exposes all three quantities at runtime, which is the practical form the contribution takes."));
col.push(body("Several limitations bound these findings. All local measurements come from one hardware configuration, so absolute figures, though not the break-even relationship, are hardware-specific. The large tier is a 7B model rather than a frontier-scale one, which caps the inter-tier energy ratio and hence achievable savings; a larger tier would raise the ceiling and we expect it to increase reported savings without changing the accounting argument. Judged quality on open-ended text is a model's assessment on an ordinal scale, mitigated but not eliminated by our second judge and human sample. Semantic entropy was evaluated at three samples for cost reasons, which quantises the measure to three values and limits its resolution; a larger sample count may recover discriminative power at proportionally higher energy cost. Finally, NVML captures accelerator power only, excluding host and cooling overheads beyond the fixed PUE factor, so our absolute figures understate total facility energy."));

// VII. Conclusion
col.push(sec("VII", "Future Work"));
col.push(body("Three directions follow. Since neither token entropy nor three-sample semantic entropy discriminates answer quality on open-ended generation at small scale, while both are informative on constrained tasks, locating the capability threshold at which self-assessment becomes reliable on free-form text is the most valuable next step, and the model-zoo methodology used here provides the instrument. Repeating the primary experiment with a frontier-scale large tier would raise the inter-tier energy ratio by an order of magnitude and test whether the break-even relationship holds at that scale. Finally, extending measurement to additional hardware generations would convert our single-configuration hardware-dependence observation into a characterised relationship, which practitioners deciding between local and hosted deployment would be able to consult directly."));
col.push(sec("VIII", "Conclusion"));
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
