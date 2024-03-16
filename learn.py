import tensorflow #matematiksel işlem yapmamızı sağlayacak kütüphanemizi import ediyoruz

ozellik_cikarma_modeli=tensorflow.keras.applications.VGG16(weights='imagenet', #özellik çıkaran modelimiz VGG16 imagenet ağırlıklarıyla üretilmiştir
                                                           include_top=False,
                                                           input_shape=(224,224,3)) #giriş görüntüsünün boyutları 224x224 iken 3 rgb (renkli görüntü) den geliyor



ozellik_cikarma_modeli.summary() #summary komutuyla bu modele ait katmanları görebiliyoruz

ozellik_cikarma_modeli.trainable=True
set_trainable=False

for layer in ozellik_cikarma_modeli.layers:
    if layer.name =='block5_conv1':        #block5_conv1 'e kadar olan katmanlar dondurulsun sadece block5_conv1 ve sonraki katmanlar eğitilebilsin
        set_trainable=True                 #dondurmasaydık çok fazla parametre eğitmemiz gerekirdi bu gereksiz ve zaman kaybı yaratırdı ayrıca kullandığımız bilgisayara çok büyük yük olurdu
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False


model=tensorflow.keras.models.Sequential() #boş bir model oluşturuyoruz. Sequential metodu katmanlı yapıyı vektör yapısına çevirmeyi sağlıyor 

model.add(ozellik_cikarma_modeli) #oluşturduğumuz boş modelin 1.katmanına bizim katmanlı olan özellik çıkarma modelini ekliyoruz

model.add(tensorflow.keras.layers.Flatten()) #boş modelin 1. katmanına eklediğimiz modeli Flatten metodu ile düzleştiriyoruz ve tek bir vektör haline getirmiş oluyoruz (yeni modelimizin 2.katmanı oldu)

model.add(tensorflow.keras.layers.Dense(512,activation='relu')) #son katmanda zaten 7*7*512=25088 vardı arkasına 256 tane nöron  daha ekliyoruz (yeni modelimizin 3.katmanı oldu)

model.add(tensorflow.keras.layers.Dense(2,activation='softmax')) #veri setinde 2 sınıfı karşılaştıracağımız için 2 nöron ekliyoruz (yeni modelimizin 4. katmanı oldu)

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(lr=1e-6),
              metrics=['acc'])  #modelimize herşeyi ekledikten sonra dışardan erişimi engelliyoruz, kapatıyoruz (compile ediyoruz)

model.summary() #summary komutuyla son oluşturduğumuz modelimizi tekrardan inceliyoruz ve yukarıda eklediğimiz katmanları görebiliyoruz


#şimdi görüntüleri verip verisetlerimizi oluşturacağız
egitim="dataset/Egitim" #projedeki eğitim klasörünün yolu
gecerleme="dataset/Gecerleme" #projedeki geçerleme klasörünün yolu
test="dataset/Test" #projedeki test klasörünün yolu

#önişleme aşaması
egitset=tensorflow.keras.preprocessing.image.ImageDataGenerator(   #burada görüntü önişleme modülüyle eğitim klasörüne attığımız resimleri eğitim için kullanmak istiyorsak bizden bazı ayarlamalar yapmamızı istiyor
    rescale=1./255, #resimlerde piksel değerleri 0-256 arasında oluyor biz bunu 0-1 arasında ayarlamak için 255'e bölüyoruz 
    rotation_range=40, #döndürme ayarlamaları
    width_shift_range=0.2, #sağa sola kaydırmalar
    height_shift_range=0.2, #yukarı aşağı kaydırmalar          #bunların hepsi birer veri artıma yöntemidir
    shear_range=0.2, #kırpmak                                    
    zoom_range=0.2, #yakınlaştırma
    horizontal_flip=True, #aynalama (resimler ters de durabilir)
    fill_mode='nearest',
    
)

#şimdi yukarıdaki önişleme aşaması eğitim veri setlerine uygulanacak 
egitim_uygula=egitset.flow_from_directory(
    egitim,
    target_size=(224,224), #bizim modelimiz boyut olarak 224x224 desteklediği için kırpıyoruz
    batch_size=16, #her seferinde 16 parça okuyacak
)

gecerleset=tensorflow.keras.preprocessing.image.ImageDataGenerator( #bu kısım geçerleme aşaması için gerekli
    rescale=1./255  #resimlerde piksel değerleri 0-256 arasında oluyor biz bunu 0-1 arasında ayarlamak için 255'e bölüyoruz 
)

#şimdi yukarıdaki işlem geçerleme veri setlerine uygulanacak
gecerleme_uygula=gecerleset.flow_from_directory(
    gecerleme,
    target_size=(224,224), #bizim modelimiz boyut olarak 224x224 desteklediği için kırpıyoruz
    batch_size=16, #her seferinde 16 parça okuyacak
)

#eğitim işlemine geldik
egitim_takip=model.fit_generator(
    egitim_uygula, #yukarıda tanımladığımız eğitim verimiz
    steps_per_epoch=10, #bir devirde(epochta) 10 kere 16 tane (batch_size) görüntü gönderecek
    epochs=2 , #2 devir işlemi gerçekleşecek
    validation_data=gecerleme_uygula, #verilerimizi yukarıda tanımladığımız değişkenden alacak
    validation_steps=1 #geçerlemeyi 1 defa yapsın
)

model.save('egitilmis_model.h5') #modeli kaybetmemek için h5 uzantısıyla kaydediyoruz

#test işlemi
test_datagen=tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255 #resimlerde piksel değerleri 0-256 arasında oluyor biz bunu 0-1 arasında ayarlamak için 255'e bölüyoruz 
)
#yukarıdaki işlem test veri setlerine uygulanacak
test_generator=test_datagen.flow_from_directory(
    test,
    target_size=(224,224),
    batch_size=16,
)

#sonuçları terminale yazdıralım
test_loss, test_acc =model.evaluate_generator(test_generator,steps=50)
print("test_acc: ",test_acc)

