"""
Conexión a la base de datos de producción (PostgreSQL en Railway).

Esta es la misma base transaccional que usa NexWearAPI (.NET) — no se usa
una base separada para los modelos, para poder entrenar y predecir sobre
datos de negocio genuinos.
"""

from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)