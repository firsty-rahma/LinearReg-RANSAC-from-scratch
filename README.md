# RANSAC-from-scratch
Ini adalah penerapan algoritma RANSAC dengan Linear Regression. Repo ini dibuat dalam rangka menyelesaikan tugas Final Project dari kelas Advanced ML di Pacmann

## Latar belakang
RANSAC (Random Sample Consensus) merupakan sebuah algoritma iteratif yang tujuannya adalah mengestimasi parameter acak dari sebuah model dengan menggunakan  data yang memiliki _outlier_ dimana _outlier_ tersebut dianggap tidak mempengaruhi nilai dari estimasi model.
RANSAC dibuat berdasarkan dua permasalahan pada estimasi parameter dalam memecahkan kasus LDP[^1].
1. Menentukan kecocokan antara data dan salah satu model yang tersedia
2. Menentukan parameter terbaik untuk parameter dari sebuah model
[^1]: LDP (_Location determination problem_) : Sebuah masalah yang bertujuan untuk mencari lokasi sebuah titik di dalam space yang relatif terhadap titik kontrol (landmark), dimana titik tersebut akan diproyeksikan ke gambar di dalam landmark tersebut

## Cara Kerja
RANSAC bekerja dengan menggunakan titik data sejumlah minimal _n_ untuk membuat sebuah subset data secara acak. Subset data tersebut digunakan sebagai parameter awal untuk inisiasi model yang akan digunakan. Setelah proses fitting dari model, proses selanjutnya adalah penghitungan _error_ dari titik data hasil prediksi dan mengecek apakah _error_ tersebut lebih rendah dari nilai _t_. Titik data yang memenuhi kriteria tersebut akan dikumpulkan menjadi sebuah subset titik _inlier_. Subset _inlier_ akan dihitung _error_ lagi untuk menentukan parameter dengan nilai _error_ terendah.

Proses penentuan subset inlier kadang membutuhkan proses iterasi sebanyak _k_. Nilai _k_ dapat ditentukan dengan menggunakan rumus di bawah ini.

> k = log(1 - z) / log(1-w^n)

Dimana <br />
_k_ : iterasi dari proses penentuan subset inlier<br />
_z_ : peluang RANSAC dalam mendapatkan minimal satu pilihan inlier<br />
_w_ : peluang jumlah inlier terpilih terhadap jumlah data input<br />

### _Component of learning_
* Optimasi (_cost function_) dalam RANSAC adalah tingkat error dari hasil prediksi model dengan input subset data
* Objective dalam RANSAC adalah meminimalkan tingkat error dari hasil prediksi tersebut
* Selain parameter dari model yang digunakan, parameter RANSAC yang bisa dioptimasi adalah _n_, _t_, _z_.

### Kelebihan dan kekurangan
Kelebihan dari RANSAC adalah membantu menentukan parameter input model dengan cepat. Namun kekurangannya RANSAC adalah sensitif terhadap perubahan karakteristik data, pengaturan parameter dari model dan RANSAC itu sendiri.

## Pseudocode
```
Input :
- n : jumlah data point minimal dalam satu sampel data
- t : batas nilai error maksimum yang dihasilkan dari sebuah model
- z : nilai peluang kesuksesan dalam mendapatkan inlier

Output : 
best_fit : model dengan parameter terbaik

d = 0 (d : jumlah inlier maksimal yang digunakan untuk memastikan bahwa parameter model fit dengan data)
k = inf (k : perkiraan jumlah iterasi)
k_done : 0 (k : jumlah iterasi yang telah dilakukan)
best_error : inf
prob_outlier = 1 - z

while k > k_done :
	init_inliers := randomly selected n data points from x
	init_model := model that fitted with initial_inliers
	if error from fitted model result < t:
		get the confirmed points to confirmed_inlier
	end if
	
	if the size of confirmed_inlier > d :
		better_model := model that fitted with confirmed_inliers
		if error from better model result < best error :
			best_fit := better_model
			best_error := error from better model
		end if
	end if
	
	w := length of confirmed_inlier / length of data
	prob_outlier = 1 - w
	k := log(1 - z) / log(1-((1 - prob_outlier)^n))
	k_done += 1
end while
```
