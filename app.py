from flask import Flask, render_template, redirect, url_for, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import click
from pathlib import Path
from io import BytesIO
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'iris_drunkenness_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)
CLASS_LABELS = {0: 'Drugged Iris', 1: 'Normal Iris'}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')

iris_model = load_model(MODEL_PATH, compile=False)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if eye_cascade.empty():
    raise RuntimeError('Failed to load OpenCV eye cascade data.')


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes)).convert('RGB')
    image = image.resize(IMG_SIZE)
    array = img_to_array(image) / 255.0
    return np.expand_dims(array, axis=0)


def contains_eye(file_bytes: bytes) -> bool:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise UnidentifiedImageError('Decoded image is empty.')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return len(eyes) > 0


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        password2 = request.form.get('password2', '')
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        elif password != password2:
            error = 'Passwords do not match.'
        elif User.query.filter_by(username=username).first():
            error = 'Username already exists.'

        if error:
            flash(error, 'danger')
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename.strip() == '':
            flash('Please select an image file to upload.', 'warning')
        elif not allowed_file(file.filename):
            flash('Allowed image types are: png, jpg, jpeg.', 'warning')
        else:
            file_bytes = file.read()
            try:
                if not contains_eye(file_bytes):
                    flash('No eye detected in the uploaded image. Please upload a clear eye image.', 'danger')
                else:
                    img_array = preprocess_image(file_bytes)
                    score = iris_model.predict(img_array)[0][0]
                    label_index = int(score > 0.5)
                    label = CLASS_LABELS[label_index]
                    confidence = score if label_index == 1 else 1 - score
                    prediction = {
                        'label': label,
                        'confidence': f'{confidence * 100:.2f}%'
                    }
                    flash('Analysis complete.', 'success')
            except UnidentifiedImageError:
                flash('The uploaded file is not a valid image.', 'danger')
            except Exception as exc:
                app.logger.exception('Prediction failed: %s', exc)
                flash('An unexpected error occurred while analyzing the image.', 'danger')

    return render_template('dashboard.html', username=current_user.username, prediction=prediction)


@app.cli.command('create-db')
def create_db():
    """Create database tables."""
    with app.app_context():
        db.create_all()
    click.echo('Database tables created.')


with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)
