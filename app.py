# app.py
"""
Flask UI for the loan prediction microservices system.

This application provides:
  - A dashboard (index page) showing loan prediction records.
  - JSON API endpoint (/loans) with all records.
  - Health check endpoint (/health).
  - Debug endpoint (/debug/loans/count) to verify DB content.

It relies on:
  - Producer  ->  Celery worker  ->  PostgreSQL  ->  Aggregator  ->  UI
"""

from flask import Flask, jsonify, render_template_string

from db import SessionLocal, LoanPredictionORM

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # allow UTF-8 JSON in responses


# -----------------------------------------------------------------------------
# HTML UI
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    """
    Render the main dashboard page.

    The page:
      - Shows high level KPI cards (total, approved, rejected, approval rate).
      - Shows global aggregated statistics from the Aggregator microservice.
      - Shows a dynamic segmentation area where the user can choose a feature
        (e.g., person_gender, loan_intent) and see segment statistics.
      - Displays a table with all loan predictions.
      - Refreshes data periodically from /loans, /aggregates and /segments.
    """
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Loan Prediction Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f5f5f5;
                margin: 0;
                padding: 20px;
            }
            h1, h2 {
                text-align: center;
                margin-top: 0;
            }
            .container {
                max-width: 1280px;
                margin: 0 auto;
                background: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            p {
                line-height: 1.5;
            }

            /* KPI cards */
            .kpi-row {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin: 20px 0;
            }
            .kpi-card {
                flex: 1 1 180px;
                background: #f9fafb;
                border-radius: 8px;
                padding: 12px 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                display: flex;
                flex-direction: column;
                justify-content: center;
                text-align: center;
            }
            .kpi-title {
                font-size: 0.9rem;
                color: #666;
            }
            .kpi-value {
                font-size: 1.4rem;
                font-weight: bold;
                margin-top: 4px;
            }

            .section-title {
                margin-top: 24px;
                margin-bottom: 8px;
                font-size: 1.2rem;
                font-weight: bold;
            }

            .error {
                color: #b00020;
                margin-top: 8px;
                font-size: 0.9rem;
            }

            /* Aggregator tables */
            .aggregator-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 16px;
                margin-top: 10px;
            }
            .agg-card {
                background: #f9fafb;
                border-radius: 8px;
                padding: 10px 12px;
                border: 1px solid #e0e0e0;
                font-size: 0.9rem;
            }
            .agg-card h3 {
                margin: 0 0 6px 0;
                font-size: 0.95rem;
                text-align: center;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.85rem;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 4px 6px;
                text-align: center;
                vertical-align: middle;
            }
            th {
                background: #f0f0f0;
                font-weight: 600;
            }

            /* Predictions table */
            .table-wrapper {
                margin-top: 12px;
                max-height: 360px;
                overflow-y: auto;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            .pred-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.85rem;
            }
            .pred-table th, .pred-table td {
                border-bottom: 1px solid #eee;
                padding: 6px 8px;
                text-align: left;
                vertical-align: top;
            }
            .pred-table th {
                position: sticky;
                top: 0;
                background: #fafafa;
                z-index: 1;
            }
            .pred-table tr:nth-child(even) {
                background: #fcfcfc;
            }
            .features-cell {
                font-size: 0.8rem;
                color: #444;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 480px;
            }
            .pill {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
                color: #fff;
            }
            .pill-approved {
                background: #16a34a;
            }
            .pill-rejected {
                background: #dc2626;
            }

            /* Segmentation */
            .seg-controls {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-top: 8px;
                margin-bottom: 8px;
                font-size: 0.9rem;
            }
            .seg-controls label {
                font-weight: 600;
            }
            .seg-controls select {
                padding: 4px 6px;
                border-radius: 4px;
                border: 1px solid #ccc;
                font-size: 0.9rem;
            }
            .seg-table-wrapper {
                margin-top: 8px;
                max-height: 260px;
                overflow-y: auto;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            .seg-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.85rem;
            }
            .seg-table th, .seg-table td {
                border-bottom: 1px solid #eee;
                padding: 4px 6px;
                text-align: center;
            }
            .seg-table th {
                background: #fafafa;
                position: sticky;
                top: 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Loan prediction system</h1>
            <p>
                Each loan request flows through the pipeline:
                <strong>Producer → RabbitMQ → Celery → ML model → Postgres → Aggregator → UI</strong>.
            </p>

            <h2>Local statistics (from /loans)</h2>
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-title">Total records (client-side)</div>
                    <div class="kpi-value" id="kpi-total">0</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Approved</div>
                    <div class="kpi-value" id="kpi-approved">0</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Rejected</div>
                    <div class="kpi-value" id="kpi-rejected">0</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Approval rate</div>
                    <div class="kpi-value" id="kpi-rate">0%</div>
                </div>
            </div>

            <div class="section-title">Aggregator service statistics (from /aggregates)</div>
            <div id="aggregator-error" class="error"></div>
            <div class="aggregator-grid">
                <div class="agg-card">
                    <h3>Predicted label distribution</h3>
                    <table id="agg-pred-table">
                        <thead>
                            <tr>
                                <th>Predicted</th>
                                <th>Count</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="agg-card">
                    <h3>True label distribution</h3>
                    <table id="agg-true-table">
                        <thead>
                            <tr>
                                <th>True label</th>
                                <th>Count</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="agg-card">
                    <h3>Predicted vs True (confusion)</h3>
                    <table id="agg-matrix-table">
                        <thead>
                            <tr id="agg-matrix-header"></tr>
                        </thead>
                        <tbody id="agg-matrix-body"></tbody>
                    </table>
                </div>
            </div>

            <div class="section-title">Dynamic segmentation</div>
            <div class="seg-controls">
                <label for="segment-feature">Segment by feature:</label>
                <select id="segment-feature">
                    <option value="person_gender">person_gender</option>
                    <option value="loan_intent">loan_intent</option>
                    <option value="person_home_ownership">person_home_ownership</option>
                    <option value="loan_grade">loan_grade</option>
                    <option value="previous_loan_defaults_on_file">previous_loan_defaults_on_file</option>
                </select>
                <span id="segment-feature-label"></span>
            </div>
            <div id="segment-error" class="error"></div>
            <div class="seg-table-wrapper">
                <table class="seg-table">
                    <thead>
                        <tr>
                            <th>Feature value</th>
                            <th>Total</th>
                            <th>Approved</th>
                            <th>Rejected</th>
                            <th>Approval rate</th>
                        </tr>
                    </thead>
                    <tbody id="segment-body"></tbody>
                </table>
            </div>

            <div class="section-title">Loan prediction table</div>
            <div id="error" class="error"></div>

            <div class="table-wrapper">
                <table class="pred-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Predicted</th>
                            <th>Confidence</th>
                            <th>True label</th>
                            <th>Features (raw JSON)</th>
                        </tr>
                    </thead>
                    <tbody id="loans-body">
                        <!-- rows injected by JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            function formatFeatures(text) {
                if (!text) return "";
                if (text.length <= 140) return text;
                return text.slice(0, 140) + "…";
            }

            function updateKpis(rows) {
                const totalEl = document.getElementById("kpi-total");
                const approvedEl = document.getElementById("kpi-approved");
                const rejectedEl = document.getElementById("kpi-rejected");
                const rateEl = document.getElementById("kpi-rate");

                const total = rows.length;
                let approved = 0, rejected = 0;
                rows.forEach(r => {
                    if (r.predicted_label === "Approved") approved++;
                    else if (r.predicted_label === "Rejected") rejected++;
                });

                totalEl.textContent = total;
                approvedEl.textContent = approved;
                rejectedEl.textContent = rejected;
                const rate = total > 0 ? (approved / total * 100).toFixed(1) : "0.0";
                rateEl.textContent = rate + "%";
            }

            async function loadLoans() {
                const errorDiv = document.getElementById("error");
                const tbody = document.getElementById("loans-body");
                errorDiv.textContent = "";
                tbody.innerHTML = "";

                try {
                    const resp = await fetch("/loans");
                    if (!resp.ok) {
                        const text = await resp.text();
                        errorDiv.textContent =
                            "Error loading /loans: HTTP " + resp.status + " - " + text;
                        console.error("HTTP error from /loans:", resp.status, text);
                        return;
                    }
                    const data = await resp.json();

                    updateKpis(data);

                    data.forEach(row => {
                        const tr = document.createElement("tr");

                        const tdId = document.createElement("td");
                        tdId.textContent = row.id;

                        const tdPred = document.createElement("td");
                        const span = document.createElement("span");
                        span.textContent = row.predicted_label || "";
                        if (row.predicted_label === "Approved") {
                            span.className = "pill pill-approved";
                        } else if (row.predicted_label === "Rejected") {
                            span.className = "pill pill-rejected";
                        }
                        tdPred.appendChild(span);

                        const tdConf = document.createElement("td");
                        tdConf.textContent =
                            row.confidence !== null && row.confidence !== undefined
                                ? Number(row.confidence).toFixed(3)
                                : "";

                        const tdTrue = document.createElement("td");
                        tdTrue.textContent = row.true_label || "";

                        const tdFeat = document.createElement("td");
                        tdFeat.textContent = formatFeatures(row.features_json || "");
                        tdFeat.className = "features-cell";
                        tdFeat.title = row.features_json || "";

                        tr.appendChild(tdId);
                        tr.appendChild(tdPred);
                        tr.appendChild(tdConf);
                        tr.appendChild(tdTrue);
                        tr.appendChild(tdFeat);
                        tbody.appendChild(tr);
                    });
                } catch (err) {
                    console.error("JavaScript error while loading /loans:", err);
                    errorDiv.textContent =
                        "JavaScript error while loading /loans. See browser console for details.";
                }
            }

            function fillSimpleAggTable(tbodyId, dataObj) {
                const tbody = document.getElementById(tbodyId);
                tbody.innerHTML = "";
                if (!dataObj || Object.keys(dataObj).length === 0) {
                    const tr = document.createElement("tr");
                    const td = document.createElement("td");
                    td.colSpan = 2;
                    td.textContent = "No data";
                    tr.appendChild(td);
                    tbody.appendChild(tr);
                    return;
                }
                Object.keys(dataObj).forEach(key => {
                    const tr = document.createElement("tr");
                    const tdKey = document.createElement("td");
                    tdKey.textContent = key;
                    const tdVal = document.createElement("td");
                    tdVal.textContent = dataObj[key];
                    tr.appendChild(tdKey);
                    tr.appendChild(tdVal);
                    tbody.appendChild(tr);
                });
            }

            function fillMatrixTable(matrix) {
                const headerRow = document.getElementById("agg-matrix-header");
                const body = document.getElementById("agg-matrix-body");
                headerRow.innerHTML = "";
                body.innerHTML = "";

                if (!matrix || Object.keys(matrix).length === 0) {
                    const th = document.createElement("th");
                    th.textContent = "No data";
                    headerRow.appendChild(th);
                    return;
                }

                // Collect all unique true labels across all predicted keys
                const trueLabelsSet = new Set();
                Object.values(matrix).forEach(inner => {
                    Object.keys(inner).forEach(t => trueLabelsSet.add(t));
                });
                const trueLabels = Array.from(trueLabelsSet).sort();

                // Build header: first empty cell, then true labels
                const thEmpty = document.createElement("th");
                thEmpty.textContent = "Predicted / True";
                headerRow.appendChild(thEmpty);
                trueLabels.forEach(t => {
                    const th = document.createElement("th");
                    th.textContent = t;
                    headerRow.appendChild(th);
                });

                // Build body rows
                Object.keys(matrix).forEach(pred => {
                    const tr = document.createElement("tr");
                    const tdLabel = document.createElement("td");
                    tdLabel.textContent = pred;
                    tr.appendChild(tdLabel);

                    trueLabels.forEach(t => {
                        const td = document.createElement("td");
                        const val =
                            matrix[pred] && Object.prototype.hasOwnProperty.call(matrix[pred], t)
                                ? matrix[pred][t]
                                : 0;
                        td.textContent = val;
                        tr.appendChild(td);
                    });
                    body.appendChild(tr);
                });
            }

            async function loadAggregates() {
                const aggErrorDiv = document.getElementById("aggregator-error");
                aggErrorDiv.textContent = "";

                try {
                    const resp = await fetch("/aggregates");
                    if (!resp.ok) {
                        const text = await resp.text();
                        aggErrorDiv.textContent =
                            "Error loading /aggregates: HTTP " + resp.status + " - " + text;
                        console.error("HTTP error from /aggregates:", resp.status, text);
                        return;
                    }
                    const data = await resp.json();

                    fillSimpleAggTable("agg-pred-tbody", data.by_predicted_label);
                    fillSimpleAggTable("agg-true-tbody", data.by_true_label);
                    fillMatrixTable(data.predicted_vs_true);
                } catch (err) {
                    console.error("JavaScript error while loading /aggregates:", err);
                    aggErrorDiv.textContent =
                        "JavaScript error while loading /aggregates. See browser console for details.";
                }
            }

            async function loadSegments() {
                const featureSelect = document.getElementById("segment-feature");
                const featureLabel = document.getElementById("segment-feature-label");
                const errorDiv = document.getElementById("segment-error");
                const tbody = document.getElementById("segment-body");

                const feature = featureSelect.value;
                errorDiv.textContent = "";
                tbody.innerHTML = "";
                featureLabel.textContent = "(feature: " + feature + ")";

                try {
                    const resp = await fetch("/segments?feature=" + encodeURIComponent(feature));
                    if (!resp.ok) {
                        const text = await resp.text();
                        errorDiv.textContent =
                            "Error loading /segments: HTTP " + resp.status + " - " + text;
                        console.error("HTTP error from /segments:", resp.status, text);
                        return;
                    }
                    const data = await resp.json();
                    const segments = data.segments || [];

                    if (!segments.length) {
                        const tr = document.createElement("tr");
                        const td = document.createElement("td");
                        td.colSpan = 5;
                        td.textContent = "No data for this feature.";
                        tr.appendChild(td);
                        tbody.appendChild(tr);
                        return;
                    }

                    segments.forEach(seg => {
                        const tr = document.createElement("tr");

                        const tdVal = document.createElement("td");
                        tdVal.textContent = seg.value;

                        const tdTotal = document.createElement("td");
                        tdTotal.textContent = seg.total;

                        const tdAppr = document.createElement("td");
                        tdAppr.textContent = seg.approved;

                        const tdRej = document.createElement("td");
                        tdRej.textContent = seg.rejected;

                        const tdRate = document.createElement("td");
                        const rate = seg.approval_rate != null
                            ? (seg.approval_rate * 100).toFixed(1) + "%"
                            : "";
                        tdRate.textContent = rate;

                        tr.appendChild(tdVal);
                        tr.appendChild(tdTotal);
                        tr.appendChild(tdAppr);
                        tr.appendChild(tdRej);
                        tr.appendChild(tdRate);
                        tbody.appendChild(tr);
                    });
                } catch (err) {
                    console.error("JavaScript error while loading /segments:", err);
                    errorDiv.textContent =
                        "JavaScript error while loading /segments. See browser console for details.";
                }
            }

            window.addEventListener("load", () => {
                // Assign tbody IDs for aggregator tables
                document.querySelector("#agg-pred-table tbody").id = "agg-pred-tbody";
                document.querySelector("#agg-true-table tbody").id = "agg-true-tbody";

                loadLoans();
                loadAggregates();
                loadSegments();

                // Refresh periodically
                setInterval(() => {
                    loadLoans();
                    loadAggregates();
                    loadSegments();
                }, 5000);

                // Reload segments when the selected feature changes
                document.getElementById("segment-feature").addEventListener("change", () => {
                    loadSegments();
                });
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Simple health check endpoint used by monitoring or load balancers.
    """
    return {"status": "ok"}, 200


# -----------------------------------------------------------------------------
# JSON API: list all loan predictions
# -----------------------------------------------------------------------------
@app.get("/loans")
def list_loans():
    """
    Return all loan prediction records from the database as JSON.

    Each element contains:
      - id
      - predicted_label
      - confidence
      - true_label
      - features_json (raw feature dictionary as JSON string)
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(LoanPredictionORM)
            .order_by(LoanPredictionORM.id.desc())
            .all()
        )
        data = [
            {
                "id": r.id,
                "predicted_label": r.predicted_label,
                "confidence": r.confidence,
                "true_label": r.true_label,
                "features_json": r.features_json,
            }
            for r in rows
        ]
        return jsonify(data), 200
    except Exception as e:
        print("[APP] Error in /loans:", repr(e))
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# -----------------------------------------------------------------------------
# Debug endpoint – count of rows (useful for quick checks)
# -----------------------------------------------------------------------------
@app.get("/debug/loans/count")
def debug_loans_count():
    """
    Return only the number of rows in the loan_predictions table.

    This endpoint is helpful to verify that:
      - The Flask app can connect to the database.
      - The table exists and contains rows.
    """
    db = SessionLocal()
    try:
        count = db.query(LoanPredictionORM).count()
        print(f"[APP] /debug/loans/count -> {count}")
        return {"count": count}, 200
    except Exception as e:
        print("[APP] Error in /debug/loans/count:", repr(e))
        return {"error": str(e)}, 500
    finally:
        db.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
