# Flask-based RESTful API for product info management
#import packages
from flask import Flask, request, jsonify
from threading import Lock  #handling concurrency

app = Flask(__name__)

#in-memory data storage
products = {}
lock = Lock()

#error handling
def error_response(message, status_code):
    return jsonify({'error': message}), status_code

#CRUD operations
#CREATE: create a new product
@app.route('/products', methods=['POST'])
def create_product():
    #get data from request
    data = request.get_json()
    if not data or 'id' not in data or 'name' not in data:
        return error_response('Missing required fields', 400)
    
    #lock concurrent transaction
    with lock:
        #check if product already exists
        product_id = data['id']
        if product_id in products:
            return error_response('Product already exists', 400)
        
        #insert data
        products[product_id] = {
            'name': data['name'],
            'description': data.get('description', ''),
            'price': data.get('price', 0),
            'quantity': data.get('quantity', 0),
        }

    return jsonify({"message": "Product created successfully"}), 201

#READ: retrieve a product
@app.route('/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    #lock concurrent transaction
    with lock:
        #get product from dict
        product = products.get(product_id)
        if not product:
            return error_response('Product not found', 404)
    
    return jsonify(product), 200

#UPDATE: update a product
@app.route('/products/<int:product_id>', methods=['PUT'])
def update_product(product_id):
    #get product from dict
    product = products.get(product_id)
    if not product:
        return error_response('Product not found', 404)
    
    #get data from request
    data = request.get_json()
    if not data:
        return error_response('Missing required fields', 400)
    
    #lock concurrent transaction
    with lock:
        #update data
        product['name'] = data.get('name', product['name'])
        product['description'] = data.get('description', product['description'])
        product['price'] = data.get('price', product['price'])
        product['quantity'] = data.get('quantity', product['quantity'])

    return jsonify({"message": "Product updated successfully"}), 200
    
#DELETE delete a product
@app.route('/products/<int:product_id>', methods=['DELETE'])
def delete_product(product_id):
    #lock concurrent transaction:
    with lock:
        #get product from dict
        product = products.get(product_id)
        if not product:
            return error_response('Product not found', 404)
        
        #remove product
        del products[product_id]
        
        return jsonify({"message": "Product deleted successfully"}), 200