1. **Установите Poetry** (если не установлен):
   ```bash
   pip install poetry
   ```

2. **Установите зависимости через Poetry**:
   ```bash
   poetry install
   ```

3. **Активируйте виртуальное окружение**:
   ```bash
   poetry shell
   ```

**Обучение**:
   ```bash
   poetry run python main.py
   ```

**Тест модели**:
   ```bash
   poetry run python test.py
   ```

**Просмотр логов**:
   ```bash
   tensorboard --logdir runs
   ```