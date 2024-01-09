import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pdfkit
from datetime import datetime
import tempfile
import os

# judul
st.title("Website Strategi Untuk Rekomendasi Penyimpanan Buku di Perpustakaan Umum Kota Cimahi")

# File upload Dataset
uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx"])

# Periksa apakah file telah diunggah
if uploaded_file is not None:
    # Load dataset from the uploaded file
    df = pd.read_excel(uploaded_file)

    # Ringkasan Dataset
    st.header("Dataset")
    st.write(df)

    # Setelan Tampilan Dataset
    st.header("Tampilan Periode")
    by_year = st.selectbox("Inputkan Berdasarkan Tahun", [
                           "Semua"] + df["Tgl Pinjam"].dt.year.unique().tolist())
    if by_year != "Semua":
        df = df[df["Tgl Pinjam"].dt.year == by_year]

    months = {
        1: 'Januari',
        2: 'Februari',
        3: 'Maret',
        4: 'April',
        5: 'Mei',
        6: 'Juni',
        7: 'Juli',
        8: 'Agustus',
        9: 'September',
        10: 'Oktober',
        11: 'November',
        12: 'Desember'
    }

    by_month = st.selectbox("Inputkan Berdasarkan Bulan", [
                            "Semua"] + list(months.values()))
    if by_month != "Semua":
        month_number = list(months.keys())[
            list(months.values()).index(by_month)]
        if not df[df["Tgl Pinjam"].dt.month == month_number].empty:
            df = df[df["Tgl Pinjam"].dt.month == month_number]
        else:
            st.error(f"Tidak ada transaksi pada bulan {by_month}")
            st.stop()
    st.write(df)

    # Total Produk dan Transaksi
    st.header("Total Kategori dan Transaksi")
    total_produk = df["Kategori"].nunique()
    total_transaksi = df["Tgl Pinjam"].nunique()
    st.write(f"Total kategori: {total_produk}")
    st.write(f"Total Transaksi: {total_transaksi}")

    # Grafik Data Peminjaman Buku
    st.header("Grafik Data Peminjaman Buku")
    grouped_data = df.groupby("Kategori")["Tgl Pinjam"].count().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Tgl Pinjam", y="Kategori", data=grouped_data)
    plt.xlabel("Jumlah Transaksi Pinjam Buku")
    plt.ylabel("Kategori")
    plt.title("Data Peminjaman Buku")
    st.pyplot(plt)

    # Input nilai min support dan min confidence
    st.header("Input Nilai Support dan Confidence")
    min_support = st.number_input(
        "Masukkan nilai min support:", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    min_confidence = st.number_input(
        "Masukkan nilai min confidence:", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

    # Analisis Apriori
    te = TransactionEncoder()
    if uploaded_file is not None:
        te_ary = te.fit_transform(df.groupby("Tgl Pinjam")["Kategori"].apply(list))
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(
            df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift")

        # Menampilkan Rules
        st.subheader("Hasil Analisis Apriori")
        if not rules.empty:
            # Menambahkan nilai minimum confidence dan lift
            rules = rules[(rules["confidence"] >= min_confidence) &
                          ((rules["lift"] > 1) | (rules["lift"] == 1))]

            if rules.empty:
                st.warning(
                    "Tidak ada rules dengan nilai Lift Ratio lebih dari 1 atau sama dengan 1.")
            else:
                # Menambahkan kolom "Rules"
                rules['Rules'] = rules.apply(
                    lambda row: f"Jika meminjam buku {', '.join(list(row['antecedents']))}, maka cenderung meminjam buku {', '.join(list(row['consequents']))}.", axis=1)
                # Menambahkan sorting berdasarkan confidence (dari yang terbesar ke terkecil)
                rules = rules.sort_values(by="confidence", ascending=False)
                st.dataframe(rules[['Rules', 'support', 'confidence', 'lift']].reset_index(drop=True))
        else:
            st.warning("Tidak ada hasil Apriori")

    # Membuat file PDF sementara hanya jika ada hasil Apriori
    if not rules.empty:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_filename = temp_pdf.name

            # Menambahkan Keterangan mengenai Support, Confidence, dan Lift
            explanation = """
           <h3>Keterangan:</h3>
            <p><strong>RULES</strong><br>Rules disini merujuk kepada aturan-aturan pola peminjaman buku dalam transaksi peminjaman buku.</p>
            <p><strong>SUPPORT</strong><br>Support yaitu ukuran yang digunakan untuk mengukur seberapa sering pasangan buku muncul bersamaan dalam transaksi peminjaman buku.</p>
            <p><strong>CONFIDANCE</strong><br>Confidence yaitu ukuran yang digunakan untuk mengukur seberapa besar peluang bahwa suatu buku akan dipinjam bersamaan dalam transaksi peminjaman buku.</p>
            <p><strong>LIFT</strong><br>Lift yaitu ukuran yang digunakan untuk mengukur seberapa kuat kemungkinan buku dipinjam oleh pengunjung.</p>
            <p><strong>Kesimpulan</strong><br>.Kesimpulannya jadi hasil rules pada kategori buku harus didekatkan pada rak perputakaan.</p>

            """

            # Menambahkan judul pada tampilan PDF
            title = "<h1>Hasil Analisis Apriori</h1>"

            # Mengatur gaya CSS untuk tabel
            table_style = """
            <style>
            table {
                border-collapse: collapse;
                width: 100%;
                font-size: 14px;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            </style>
            """

            # Menggabungkan semua elemen ke dalam satu HTML
            html = f"{title}{table_style}{rules[['Rules', 'support', 'confidence', 'lift']].reset_index(drop=True).to_html(index=False)}{explanation}"

            # Menentukan path ke file executable wkhtmltopdf
            path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
            config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

            # Menyimpan laporan PDF
            pdfkit.from_string(html, temp_filename, configuration=config)

            # Mengunduh file PDF
            with open(temp_filename, "rb") as f:
                st.download_button(label="Unduh Laporan", data=f, file_name="Hasil_Analisis_Apriori.pdf", mime="application/pdf")

        # Menghapus file sementara setelah diunduh
        os.remove(temp_filename)
