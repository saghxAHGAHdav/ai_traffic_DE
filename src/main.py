import sys
import json
import os
import random
import datetime
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from design import Ui_MainWindow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

setka = 32
media = os.path.join(os.path.dirname(__file__), "media")
items = {
    "btnblue": ("синяя машина", "BCvertical.png"),
    "btnred": ("красная машина", "Cvertical.png"),
    "proad": ("перекресток", "Rcrossroads.png"),
    "vroad": ("дорога вертик.", "Rvertical.png"),
    "btnstop": ("знак 'стоп'", "Stop.png"),
    "btnpr": ("знак 'движение прямо'", "Start.png"),
    "vert": ("зебра вертик.", "Zvertical.png"),
    "hor": ("зебра горизонт.", "Zhorizontal.png"),
    "svred": ("светофор красный", "TLred.png"),
    "svgr": ("светофор зеленый", "TLgreen.png"),
    "svyel": ("светофор желтый", "TLyellow.png"),
    "btnped": ("пешеход", "Pedestrain.png"),
}
z_val = {"TL": 3, "Start": 3, "Stop": 3, "BC": 2, "C": 2, "Pedestrain": 2, "Z": 1, "R": 0}

class GridModel:
    def __init__(self):
        self.grid = {}

    def update(self, items):
        self.grid.clear()
        for it in items:
            if hasattr(it, 'obj_id') and ('дорога' in it.name or 'перекресток' in it.name):
                self.grid[(int(it.x() / setka), int(it.y() / setka))] = {
                    'type': it.name, 'rotation': it.current_rotation
                }

class CanvasObject(QGraphicsPixmapItem):
    def __init__(self, obj_id, name, img_name, pixmap):
        super().__init__(pixmap)
        self.obj_id = obj_id
        self.name = name
        self.img_name = img_name
        self.current_rotation = 0
        self.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(next((v for k, v in z_val.items() if k in img_name), 0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            return QPointF(round(value.x() / setka) * setka, round(value.y() / setka) * setka)
        return super().itemChange(change, value)

    def rotate_90(self):
        self.current_rotation = (self.current_rotation + 90) % 360
        self.setRotation(self.current_rotation)

class CarDriver:
    def __init__(self, car_item, grid_map):
        self.car = car_item
        self.grid_map = grid_map
        self.moving = False
        self.stopped = False
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.animate)
        self.anim_progress = 0
        self.anim_duration = 400
        self.anim_start_pos = None
        self.anim_end_pos = None
        self.anim_start_rot = 0
        self.anim_end_rot = 0
        self.direction = int(car_item.current_rotation / 90) % 4
        self.prev_cell = None

    def get_cell(self):
        return (int(self.car.x() / setka), int(self.car.y() / setka))

    def get_cell_type(self, x, y):
        if (x, y) not in self.grid_map: return None
        return self.grid_map[(x, y)].get('type')

    def get_cell_rotation(self, x, y):
        if (x, y) not in self.grid_map: return 0
        return self.grid_map[(x, y)].get('rotation', 0)

    def is_road_cell(self, x, y, direction):
        cell_type = self.get_cell_type(x, y)
        if cell_type is None: return False
        if 'перекресток' in cell_type: return True
        if 'дорога' in cell_type:
            cell_rot = self.get_cell_rotation(x, y)
            if cell_rot % 180 == 0: return direction in [0, 2]
            else: return direction in [1, 3]
        return False

    def find_road_exit(self, x, y, direction, max_steps=6):
        dx_dy = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dx_dy[direction]
        cx, cy = x, y
        for _ in range(max_steps):
            cx += dx
            cy += dy
            cell_type = self.get_cell_type(cx, cy)
            if cell_type is None: return None
            if 'перекресток' in cell_type: continue
            if 'дорога' in cell_type:
                cell_rot = self.get_cell_rotation(cx, cy)
                if cell_rot % 180 == 0:
                    if direction in [0, 2]: return cx, cy
                else:
                    if direction in [1, 3]: return cx, cy
                return None
            if 'машина' in cell_type: return None
        return None

    def find_next_move(self):
        x, y = self.get_cell()
        direction = self.direction
        dx_dy = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = dx_dy[direction]
        new_x, new_y = x + dx, y + dy
        next_type = self.get_cell_type(new_x, new_y)
        if next_type and 'дорога' in next_type:
            if self.is_road_cell(new_x, new_y, direction): return new_x, new_y, direction
        if next_type and 'перекресток' in next_type: return new_x, new_y, direction
        for try_dir in [direction, (direction + 1) % 4, (direction - 1) % 4]:
            result = self.find_road_exit(x, y, try_dir)
            if result is not None:
                if self.prev_cell and (result[0], result[1]) == self.prev_cell: continue
                return result[0], result[1], try_dir
        return None

    def start_moving(self):
        if self.moving or self.stopped: return
        result = self.find_next_move()
        if result is None:
            self.stopped = True
            return
        new_x, new_y, new_direction = result
        self.anim_start_pos = QPointF(self.car.x(), self.car.y())
        self.anim_end_pos = QPointF(new_x * setka, new_y * setka)
        self.anim_start_rot = self.car.current_rotation
        self.anim_end_rot = new_direction * 90
        self.anim_progress = 0
        self.moving = True
        self.anim_timer.start(16)

    def animate(self):
        self.anim_progress += 16 / self.anim_duration
        if self.anim_progress >= 1.0:
            self.anim_progress = 1.0
            self.anim_timer.stop()
            self.moving = False
            self.prev_cell = self.get_cell()
            self.car.setPos(self.anim_end_pos)
            self.car.current_rotation = self.anim_end_rot
            self.car.setRotation(self.anim_end_rot)
            self.direction = int(self.anim_end_rot / 90) % 4
            QTimer.singleShot(100, self.start_moving)
            return
        x = self.anim_start_pos.x() + (self.anim_end_pos.x() - self.anim_start_pos.x()) * self.anim_progress
        y = self.anim_start_pos.y() + (self.anim_end_pos.y() - self.anim_start_pos.y()) * self.anim_progress
        self.car.setPos(x, y)
        diff = self.anim_end_rot - self.anim_start_rot
        if diff > 180: diff -= 360
        if diff < -180: diff += 360
        new_rot = self.anim_start_rot + diff * self.anim_progress
        self.car.setRotation(new_rot)

class WorkCanvas(QGraphicsView):
    def __init__(self, scene, main_window, parent=None):
        super().__init__(scene, parent)
        self.main_window = main_window
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def mousePressEvent(self, event):
        mw = self.main_window
        pos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton and mw.placement_mode:
            snap = QPointF(round(pos.x() / setka) * setka, round(pos.y() / setka) * setka)
            mw.create_object(mw.current_item_name, mw.current_img_name, snap)
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("перекресток")
        self.object_counter = 0
        self.placement_mode = False
        self.current_item_name = self.current_img_name = None
        self.active_btn = None
        self.drivers = []
        self.grid_model = GridModel()
        self.model = None
        self.current_epoch = 0
        self.scene_setup()
        self.setup_tables()
        self.connect_signals()
        self.scene.selectionChanged.connect(self.update_properties)

    def setup_tables(self):
        self.loss_table = QTableWidget(0, 3, self.centralwidget)
        self.loss_table.setGeometry(self.loss.geometry())
        self.loss_table.setHorizontalHeaderLabels(["эпоха", "потери", "точность"])
        self.loss_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.loss.deleteLater()
        self.loss = self.loss_table

        self.log_table = QTableWidget(0, 3, self.centralwidget)
        self.log_table.setGeometry(self.log.geometry())
        self.log_table.setHorizontalHeaderLabels(["время", "событие", "детали"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log.deleteLater()
        self.log = self.log_table

        self.svoistva.setColumnCount(2)
        self.svoistva.setHorizontalHeaderLabels(["свойство", "значение"])
        self.svoistva.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def scene_setup(self):
        self.scene = QGraphicsScene(0, 0, 1600, 1200)
        pix = QPixmap(setka, setka)
        pix.fill(Qt.white)
        p = QPainter(pix)
        p.setPen(QColor(200, 200, 200, 80))
        p.drawRect(0, 0, setka, setka)
        p.end()
        self.scene.setBackgroundBrush(QBrush(pix))
        old_view = self.graphicsView
        self.graphicsView = WorkCanvas(self.scene, self, self.centralwidget)
        self.graphicsView.setGeometry(old_view.geometry())
        old_view.deleteLater()

    def connect_signals(self):
        for btn_name, (name, img) in items.items():
            btn = getattr(self, btn_name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, n=name, i=img, b=btn: self.toggle(n, i, b))
        self.btn_opt.clicked.connect(self.rotate_obj)
        self.btndel.clicked.connect(self.del_obj)
        self.learn.clicked.connect(self.train_one_epoch)
        self.action_JSON.triggered.connect(self.save_json)
        self.action_JSON_2.triggered.connect(self.load_json)

    def toggle(self, name, img, btn):
        if self.active_btn == btn:
            self.active_btn.setChecked(False)
            self.active_btn = None
            self.placement_mode = False
        else:
            if self.active_btn: self.active_btn.setChecked(False)
            self.active_btn = btn
            self.placement_mode = True
            self.current_item_name = name
            self.current_img_name = img

    def create_object(self, name, img_name, pos):
        self.object_counter += 1
        pix = QPixmap(os.path.join(media, img_name))
        if pix.isNull():
            pix = QPixmap(setka, setka)
            pix.fill(Qt.gray)
        obj = CanvasObject(f"id_{self.object_counter}", f"{name} #{self.object_counter}", img_name, pix)
        obj.setPos(round((pos.x() - pix.width() / 2) / setka) * setka, round((pos.y() - pix.height() / 2) / setka) * setka)
        self.scene.addItem(obj)
        self.scene.clearSelection()
        obj.setSelected(True)
        self.grid_model.update(self.scene.items())
        return obj

    def del_obj(self):
        for item in self.scene.selectedItems():
            if isinstance(item, CanvasObject): self.scene.removeItem(item)
        self.grid_model.update(self.scene.items())
        self.update_properties()

    def rotate_obj(self):
        for item in self.scene.selectedItems():
            if isinstance(item, CanvasObject): item.rotate_90()
        self.grid_model.update(self.scene.items())
        self.update_properties()

    def update_properties(self):
        self.svoistva.setRowCount(0)
        selected = self.scene.selectedItems()
        if not selected: return
        item = selected[0]
        if not isinstance(item, CanvasObject): return
        self.svoistva.insertRow(0)
        self.svoistva.setItem(0, 0, QTableWidgetItem("объект"))
        self.svoistva.setItem(0, 1, QTableWidgetItem(item.name))
        self.svoistva.insertRow(1)
        self.svoistva.setItem(1, 0, QTableWidgetItem("угол"))
        self.svoistva.setItem(1, 1, QTableWidgetItem(f"{item.current_rotation}°"))

    def get_grid_features_and_target(self):
        cars_count = sum(1 for it in self.scene.items() if isinstance(it, CanvasObject) and 'машина' in it.name)
        peds_count = sum(1 for it in self.scene.items() if isinstance(it, CanvasObject) and 'пешеход' in it.name)
        lights_count = sum(1 for it in self.scene.items() if isinstance(it, CanvasObject) and 'светофор' in it.name)
        temperature = peds_count * -10
        light_binary = 1 if lights_count > 0 else 0
        cars = cars_count
        weather_factor = 1.5 if (temperature < -10 or temperature > 30) else 1.0
        light_factor = 1.2 if light_binary == 1 else 0.7
        effective_cars = cars * weather_factor * light_factor
        if effective_cars < 20: traffic_level = 0
        elif effective_cars < 50: traffic_level = 1
        else: traffic_level = 2
        return [cars, temperature, light_binary], traffic_level

    def create_nn_model(self):
        model = Sequential([
            Dense(16, activation='relu', input_shape=(3,)),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_one_epoch(self):
        start_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_event(start_time, "начало обучения", f"эпоха {self.current_epoch + 1}")
        base_X, base_Y = self.get_grid_features_and_target()
        labels = {0: "1(свободно)", 1: "2(средне)", 2: "3(пробка)"}
        self.log_event(datetime.datetime.now().strftime("%H:%M:%S"), "предсказание", labels.get(base_Y, str(base_Y)))
        X_batch, Y_batch = [], []
        for _ in range(100):
            c = max(0, base_X[0] + random.randint(-3, 3))
            t = base_X[1] + random.randint(-5, 5)
            l = base_X[2]
            wf = 1.5 if (t < -10 or t > 30) else 1.0
            lf = 1.2 if l == 1 else 0.7
            eff = c * wf * lf
            lvl = 0 if eff < 20 else (1 if eff < 50 else 2)
            X_batch.append([c, t, l])
            Y_batch.append(lvl)
        X = np.array(X_batch, dtype=float)
        Y = np.array(Y_batch, dtype=int)
        X[:, 0] = X[:, 0] / 80.0
        X[:, 1] = (X[:, 1] + 30) / 70.0
        Y_cat = to_categorical(Y, num_classes=3)
        if self.model is None:
            self.model = self.create_nn_model()
            self.log_event(start_time, "создание модели", "инициализация архитектуры нс")
        history = self.model.fit(X, Y_cat, epochs=1, verbose=0)
        self.current_epoch += 1
        loss_val = history.history['loss'][0]
        acc_val = history.history['accuracy'][0]
        self.update_loss_table(self.current_epoch, loss_val, acc_val)
        self.log_event(datetime.datetime.now().strftime("%H:%M:%S"), "окончание эпохи", f"loss: {loss_val:.4f}, acc: {acc_val:.4f}")
        self.model.save("traffic_model.keras")
        self.log_event(datetime.datetime.now().strftime("%H:%M:%S"), "сохранение модели", "traffic_model.keras")

    def update_loss_table(self, epoch, loss, acc):
        row = self.loss.rowCount()
        self.loss.insertRow(row)
        self.loss.setItem(row, 0, QTableWidgetItem(str(epoch)))
        self.loss.setItem(row, 1, QTableWidgetItem(f"{loss:.4f}"))
        self.loss.setItem(row, 2, QTableWidgetItem(f"{acc:.4f}"))
        self.loss.scrollToBottom()

    def log_event(self, time_str, event, details):
        row = self.log.rowCount()
        self.log.insertRow(row)
        self.log.setItem(row, 0, QTableWidgetItem(time_str))
        self.log.setItem(row, 1, QTableWidgetItem(event))
        self.log.setItem(row, 2, QTableWidgetItem(details))
        self.log.scrollToBottom()

    def ai_on(self):
        self.grid_model.update(self.scene.items())
        grid_map = self.grid_model.grid
        self.stop_driving()
        for item in self.scene.items():
            if isinstance(item, CanvasObject) and "машина" in item.name:
                driver = CarDriver(item, grid_map)
                self.drivers.append(driver)
                driver.start_moving()

    def stop_driving(self):
        for driver in self.drivers: driver.anim_timer.stop()
        self.drivers.clear()

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "сохранить", "", "JSON (*.json)")
        if not path: return
        data = {
            "objects": {
                item.obj_id: {
                    "name": item.name, "file": item.img_name,
                    "rotation": item.current_rotation,
                    "position_x": item.x(), "position_y": item.y()
                } for item in self.scene.items() if isinstance(item, CanvasObject)
            }
        }
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "открыть", "", "JSON (*.json)")
        if not path: return
        for item in list(self.scene.items()):
            if isinstance(item, CanvasObject): self.scene.removeItem(item)
        self.stop_driving()
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            objects_data = data.get("objects", data)
            max_id = 0
            for obj_id, info in objects_data.items():
                pos = QPointF(info["position_x"], info["position_y"])
                obj = self.create_object(info["name"].split(" #")[0], info["file"], pos)
                obj.setPos(pos)
                obj.current_rotation = info.get("rotation", 0)
                obj.setRotation(obj.current_rotation)
                try: max_id = max(max_id, int(obj_id.split("_")[1]))
                except (ValueError, IndexError): pass
            self.object_counter = max_id
            self.grid_model.update(self.scene.items())
        except Exception as e:
            QMessageBox.critical(self, "ошибка", f"не удалось загрузить файл:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())