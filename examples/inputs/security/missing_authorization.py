from flask import Flask, jsonify, session

app = Flask(__name__)

@app.route("/api/invoices/<int:invoice_id>")
def get_invoice(invoice_id: int):
    if "user_id" not in session:
        return jsonify({"error": "not authenticated"}), 401
    invoice = db.query("SELECT * FROM invoices WHERE id = ?", (invoice_id,))
    return jsonify(invoice)
