import os

#File Name : config_template.py


TEMPLATE = """
    Customer Name: {customer_name}
    Customer ID: {customer_id}
    Billing Account: {bill_account}

    Purchase Order Number: {po_number}

    Billing Address: {bill_address}
    Service Address: {service_address}

    Invoice Number: {invoice_number}
    Invoice Date: {date}
    Invoice Amount: {invoice_amount}
    Invoice Currency: {currency}

    {circuit_details}
    """