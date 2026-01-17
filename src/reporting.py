import json
import os
import datetime
from typing import List, Dict, Any

class HTMLReportGenerator:
    """
    Generates a standalone HTML report with charts and tables for evaluation results.
    Uses Tailwind CSS for styling and Chart.js for visualization (via CDN).
    """
    
    @staticmethod
    def generate_report(results: List[Dict[str, Any]], output_path: str, project_name: str = "default"):
        """
        Generates the HTML file.
        """
        # 1. Calculate Statistics
        total_items = len(results)
        metrics_sum = {}
        metrics_count = {}
        layer_sum = {}
        layer_count = {}
        
        for item in results:
            if not item: continue
            # Flatten metrics from categories
            for category, data in item.items():
                if category in ["input", "meta_evaluation"]: continue
                
                # Track Layer Score
                if isinstance(data, dict):
                    l_score = data.get("final_score", 0)
                    layer_sum[category] = layer_sum.get(category, 0) + l_score
                    layer_count[category] = layer_count.get(category, 0) + 1

                if isinstance(data, dict) and "evaluators" in data:
                    for metric, result_obj in data["evaluators"].items():
                        # Handle both scalar scores (float) and complex results (dict with 'score')
                        if isinstance(result_obj, dict):
                            val = result_obj.get("score", 0)
                        else:
                            val = result_obj
                        
                        # Ensure val is a number
                        if not isinstance(val, (int, float)):
                            val = 0
                            
                        metrics_sum[metric] = metrics_sum.get(metric, 0) + val
                        metrics_count[metric] = metrics_count.get(metric, 0) + 1

        avg_scores = {k: round(v / metrics_count[k], 2) for k, v in metrics_sum.items()}
        avg_layers = {f"Layer: {k}": round(v / layer_count[k], 2) for k, v in layer_sum.items()}
        
        # Global Final Score
        global_score = round(sum(avg_layers.values()) / len(avg_layers), 2) if avg_layers else 0.0

        # Combine for charts
        display_stats = {**avg_layers, **avg_scores}
        
        # Prepare Chart Data
        chart_labels = list(avg_scores.keys())
        chart_values = list(avg_scores.values())

        # 2. Build HTML Content using f-strings
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - {project_name}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #f3f4f6; }}
        .card {{ background: white; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); }}
        summary::-webkit-details-marker {{ display: none; }}
    </style>
</head>
<body class="p-8">
    <div class="max-w-7xl mx-auto space-y-8">
        
        <!-- Header -->
        <div class="flex justify-between items-center">
            <div>
                <h1 class="text-3xl font-bold text-gray-900">Evaluation Report</h1>
                <p class="text-gray-500">Project: <span class="font-mono text-blue-600">{project_name}</span> | Items: {total_items} | Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            <a href="output_results.jsonl" class="text-blue-500 hover:text-blue-700 underline">View Raw JSON</a>
        </div>

        <!-- Metrics Dashboard -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Chart -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-4 text-gray-800">Metrics Overview (Average Tests)</h2>
                <canvas id="metricsChart"></canvas>
            </div>
            
            <!-- Summary Stats -->
            <div class="space-y-6">
                <!-- Global Score Card -->
                <div class="card border-l-4 border-blue-600">
                    <h2 class="text-lg font-semibold text-gray-700">Global Final Score</h2>
                    <p class="text-4xl font-bold text-blue-900 mt-2">{global_score}</p>
                    <p class="text-sm text-gray-500">Aggregated across {len(avg_layers)} layers and {total_items} items.</p>
                </div>
                
                <div class="card">
                    <details open class="group">
                        <summary class="text-xl font-semibold mb-4 text-gray-800 cursor-pointer list-none flex items-center justify-between">
                            Layer Performance
                            <span class="transform group-open:rotate-180 transition-transform">▼</span>
                        </summary>
                        <div class="grid grid-cols-2 gap-4">
                            {HTMLReportGenerator._render_stat_cards(avg_layers)}
                        </div>
                    </details>
                </div>

                <div class="card">
                    <details class="group">
                        <summary class="text-xl font-semibold mb-4 text-gray-800 cursor-pointer list-none flex items-center justify-between">
                            Metric Breakdown
                            <span class="transform group-open:rotate-180 transition-transform">▼</span>
                        </summary>
                        <div class="grid grid-cols-2 gap-4 mt-4">
                            {HTMLReportGenerator._render_stat_cards(avg_scores)}
                        </div>
                    </details>
                </div>

                <!-- Legend -->
                <div class="flex items-center space-x-4 text-sm text-gray-500 justify-end">
                    <span>Performance Legend:</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-green-600 mr-1"></span> > 0.7 (Good)</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-yellow-600 mr-1"></span> 0.4 - 0.7 (Fair)</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-red-600 mr-1"></span> < 0.4 (Poor)</span>
                </div>
            </div>
        </div>

        <!-- Detailed Results Table -->
        <div class="card overflow-hidden">
            <h2 class="text-xl font-semibold mb-4 text-gray-800">Detailed Results</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-1/3">Ground Truth</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-1/3">Candidate</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Scores</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {HTMLReportGenerator._render_table_rows(results)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Chart Script -->
    <script>
        const ctx = document.getElementById('metricsChart');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'Average Score',
                    data: {json.dumps(chart_values)},
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: true, max: 1.0 }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[Report] HTML report generated at {output_path}")

    @staticmethod
    def _render_stat_cards(avg_scores):
        html = ""
        for metric, score in avg_scores.items():
            color_class = "text-green-600" if score > 0.7 else "text-yellow-600" if score > 0.4 else "text-red-600"
            html += f"""
            <div class="p-4 bg-gray-50 rounded-lg">
                <p class="text-sm text-gray-500 uppercase">{metric}</p>
                <p class="text-2xl font-bold {color_class}">{score}</p>
            </div>
            """
        return html

    @staticmethod
    def _render_table_rows(results):
        html = ""
        for idx, item in enumerate(results):
            if not item: continue
            
            # Extract Input
            inp = item.get("input", {})
            gt = inp.get("ground_truth", "N/A")
            cand = inp.get("candidate", "N/A")
            item_id = inp.get("id", str(idx))
            
            # Format Scores
            scores_html = ""
            for category, data in item.items():
                if category in ["input", "meta_evaluation"]: continue
                
                # Display Layer Header
                if isinstance(data, dict):
                     layer_score = data.get("final_score", 0)
                     scores_html += f"<div class='font-bold text-gray-800 mt-2 mb-1 border-b pb-0.5'>{category} ({round(layer_score, 2)})</div>"

                if isinstance(data, dict) and "evaluators" in data:
                    for m, result_obj in data["evaluators"].items():
                        # Extract score if dict
                        if isinstance(result_obj, dict):
                            val = result_obj.get("score", 0)
                        else:
                            val = result_obj
                            
                        # Ensure numeric
                        if not isinstance(val, (int, float)):
                            val = 0
                            
                        scores_html += f"<div class='text-xs'><span class='font-medium'>{m}:</span> {round(val, 2)}</div>"

            # Truncate text
            gt_short = (gt[:100] + '...') if len(gt) > 100 else gt
            cand_short = (cand[:100] + '...') if len(cand) > 100 else cand

            html += f"""
            <tr class="hover:bg-gray-50">
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item_id}</td>
                <td class="px-6 py-4 text-sm text-gray-500" title="{gt.replace('"', '&quot;')}">{gt_short}</td>
                <td class="px-6 py-4 text-sm text-gray-500 text-blue-600" title="{cand.replace('"', '&quot;')}">{cand_short}</td>
                <td class="px-6 py-4 text-sm text-gray-500 space-y-1">{scores_html}</td>
            </tr>
            """
        return html
