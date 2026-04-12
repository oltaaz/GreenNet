from docx import Document


PATH = "/Users/oltazagraxha/Desktop/GreenNet/output/doc/RIT_Digital_Institutional_GreenNet_reorganized.docx"


doc = Document(PATH)

updates = {
    21: "GreenNet is a simulation-based framework for studying whether energy-aware link-state control can reduce network energy use without causing unacceptable degradation in Quality of Service (QoS). The project combines a graph-based network simulator, configurable traffic scenarios, an All-On controller, a heuristic controller, and a PPO-based hybrid controller within one benchmarkable environment.",
    40: "To implement and compare multiple control strategies, including an All-On controller, a heuristic controller, and a PPO-based hybrid controller that combines learned action proposals with rule-based safety constraints.",
    41: "To evaluate network behavior using both sustainability-related and performance-related metrics, including energy consumption, delivered traffic, dropped traffic, average delay, path latency, carbon-related output, and QoS compliance.",
    54: "How do AI-based approaches compare with the All-On controller and the heuristic controller under different traffic and topology conditions?",
    68: "Simulation methodology is equally important. Network simulators are valuable because they allow repeated experiments under controlled conditions, but the right level of fidelity depends on the research question. For GreenNet, the priority is not full protocol emulation; it is transparent comparison among controllers, fast iteration over seeds and scenarios, and direct integration with Python-based evaluation code. That is why the project uses a lightweight graph-based simulator rather than a heavier discrete-event platform, while still following the networking-research principle that simulation assumptions should be explicit and experiments repeatable (Henderson et al., 2008).",
    80: "Finally, the methodology includes reviewer-facing reporting. GreenNet aggregates run data into comparison tables, stored summaries, and interface views so that policy trade-offs can be inspected from the same evidence used in the written analysis. The method is therefore both computational and documentary: it is designed to produce results and to make those results auditable.",
    89: "A backend API layer connects the stored and computed results to the interface. In practice, this service exposes run lists, summaries, aggregate comparisons, and simulator-facing metadata so that the reporting surface is driven by the same data used in the written analysis.",
    101: "Implementation therefore serves the central thesis purpose directly: it turns the question of energy-aware control into a benchmarkable system with reproducible artifacts, comparable controller outputs, and explicit QoS-constrained interpretation.",
    116: "The AI stack is intentionally narrow. Stable-Baselines3 provides the PPO implementation used for the learned controller, while PyTorch supports auxiliary ML code in the repository. Scikit-learn appears in exploratory support paths, but the thesis does not depend on presenting a broad machine-learning pipeline; its main AI claim rests on the implemented PPO-based hybrid controller.",
    123: "The main conclusion from Figure 1 is conservative. The final benchmark does not confirm the thesis hypothesis, because the observed energy gains are modest relative to the acceptance target and the heuristic controller remains the strongest overall policy in aggregate reporting. At the same time, the policy labeled PPO remains QoS-acceptable under the project thresholds and performs as the strongest AI-based controller, which means the benchmark supports cautious promise rather than a headline breakthrough.",
    127: "Taken together, Figures 1 and 2 support a precise interpretation. The aggregate benchmark shows that the heuristic controller remains the strongest overall policy and that the hypothesis is not achieved. The case study shows that the AI-based controller is not decorative; it can produce beneficial trade-offs under selected conditions. The correct academic conclusion is therefore mixed: GreenNet demonstrates a real but limited AI signal inside a stronger comparative framework whose main contribution is methodological.",
    139: "The working expectation for GreenNet was that adaptive control would reduce energy use relative to the All-On controller while staying within acceptable QoS bounds. The final benchmark partially supports that direction of travel, but it does not deliver the level of improvement required for hypothesis confirmation.",
    165: "Appendix Figure A2 provides a compact scenario-by-scenario view of how the PPO-based hybrid controller changes energy use and delivered traffic relative to the All-On controller in the official final benchmark.",
    166: "Appendix Figure A2. Scenario-by-scenario PPO-based hybrid controller trade-off relative to the All-On controller in the official final benchmark.",
}

for index, text in updates.items():
    doc.paragraphs[index].text = text

doc.save(PATH)
