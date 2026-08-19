"""
ETL Pipeline — NexWear AI
==========================
Extrae datos reales de la base de producción (PostgreSQL en Railway),
detecta y reporta problemas de calidad (nulos, duplicados, atípicos),
los limpia, y guarda datasets curados en /data como almacén estructurado.

Fuente de datos: PostgreSQL (Railway) — misma base transaccional de NexWear.
Justificación: se usa la base real de producción en lugar de un dataset externo
porque el objetivo es entrenar modelos de IA sobre el comportamiento real del
negocio (ventas, catálogo, reseñas), no sobre datos genéricos de terceros.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from db import engine
from sqlalchemy import text

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log_section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def report_quality(df: pd.DataFrame, name: str) -> dict:
    """Detecta y reporta nulos, duplicados y filas vacías antes de limpiar."""
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    duplicates = df.duplicated().sum()

    print(f"\n--- Reporte de calidad: {name} ---")
    print(f"Filas totales: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    if len(nulls) > 0:
        print("Valores nulos por columna:")
        print(nulls.to_string())
    else:
        print("Sin valores nulos.")
    print(f"Filas duplicadas: {duplicates}")

    return {"nulls_by_column": nulls.to_dict(), "duplicates": int(duplicates)}


# ── EXTRACT ──────────────────────────────────────────────────────
def extract():
    log_section("EXTRACT — Extrayendo datos reales de PostgreSQL (Railway)")
    with engine.connect() as conn:
        products = pd.read_sql(text('''
            SELECT "Id", "Name", "Category", "Price", "Description", "ImageUrl", "IsActive"
            FROM "Products"
        '''), conn)

        orders = pd.read_sql(text('''
            SELECT "Id", "UserId", "Total", "Status", "CreatedAt"
            FROM "Orders"
        '''), conn)

        order_items = pd.read_sql(text('''
            SELECT "Id", "OrderId", "ProductId", "VariantId", "Quantity"
            FROM "OrderItems"
        '''), conn)

        reviews = pd.read_sql(text('''
            SELECT "Id", "UserId", "ProductId", "Rating", "Comment", "IsApproved", "CreatedAt"
            FROM "Reviews"
        '''), conn)

    print(f"Products: {len(products)} filas")
    print(f"Orders: {len(orders)} filas")
    print(f"OrderItems: {len(order_items)} filas")
    print(f"Reviews: {len(reviews)} filas")

    return products, orders, order_items, reviews


# ── TRANSFORM (limpieza) ────────────────────────────────────────
def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    log_section("TRANSFORM — Limpieza: Products")
    report_quality(df, "Products (crudo)")

    df = df.copy()
    # Normalizar categorías inconsistentes detectadas manualmente (ej. "exterior" vs "exteriores")
    df["Category"] = df["Category"].str.strip().str.lower()
    df["Category"] = df["Category"].replace({"exterior": "exteriores", "tops": "mujer"})

    # Eliminar duplicados exactos
    before = len(df)
    df = df.drop_duplicates(subset=["Id"])
    print(f"Duplicados eliminados: {before - len(df)}")

    # Precios nulos o negativos son datos corruptos -> se excluyen
    before = len(df)
    df = df[df["Price"].notna() & (df["Price"] > 0)]
    print(f"Filas con precio nulo/negativo eliminadas: {before - len(df)}")

    # Descripción nula -> se rellena con string vacío (no es un dato crítico para el modelo)
    df["Description"] = df["Description"].fillna("")

    report_quality(df, "Products (limpio)")
    return df


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    log_section("TRANSFORM — Limpieza: Orders")
    report_quality(df, "Orders (crudo)")

    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["Id"])
    print(f"Duplicados eliminados: {before - len(df)}")

    # Total nulo o negativo es un dato corrupto
    before = len(df)
    df = df[df["Total"].notna() & (df["Total"] >= 0)]
    print(f"Filas con total inválido eliminadas: {before - len(df)}")

    report_quality(df, "Orders (limpio)")
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    log_section("TRANSFORM — Limpieza: Reviews")
    report_quality(df, "Reviews (crudo)")

    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["Id"])
    print(f"Duplicados eliminados: {before - len(df)}")

    # Comentarios nulos o vacíos no aportan a los modelos de NLP -> se marcan aparte
    df["Comment"] = df["Comment"].fillna("")
    empty_comments = (df["Comment"].str.strip() == "").sum()
    print(f"Reseñas sin comentario de texto: {empty_comments} (se conservan para rating, se excluyen del análisis NLP)")

    # Rating fuera de rango 1-5 es un dato atípico/corrupto
    before = len(df)
    df = df[df["Rating"].between(1, 5)]
    print(f"Filas con rating fuera de rango eliminadas: {before - len(df)}")

    report_quality(df, "Reviews (limpio)")
    return df


# ── LOAD ─────────────────────────────────────────────────────────
def load(products, orders, order_items, reviews):
    log_section("LOAD — Guardando datasets limpios en /data")
    products.to_csv(os.path.join(OUTPUT_DIR, "products_clean.csv"), index=False)
    orders.to_csv(os.path.join(OUTPUT_DIR, "orders_clean.csv"), index=False)
    order_items.to_csv(os.path.join(OUTPUT_DIR, "order_items_clean.csv"), index=False)
    reviews.to_csv(os.path.join(OUTPUT_DIR, "reviews_clean.csv"), index=False)
    print(f"Datasets guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    products, orders, order_items, reviews = extract()

    products_clean = clean_products(products)
    orders_clean = clean_orders(orders)
    reviews_clean = clean_reviews(reviews)

    load(products_clean, orders_clean, order_items, reviews_clean)

    log_section("ETL COMPLETO")
    print("Resumen final:")
    print(f"  Products: {len(products)} → {len(products_clean)} filas limpias")
    print(f"  Orders: {len(orders)} → {len(orders_clean)} filas limpias")
    print(f"  Reviews: {len(reviews)} → {len(reviews_clean)} filas limpias")