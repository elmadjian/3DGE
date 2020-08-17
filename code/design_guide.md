### Carregando Vídeo -> processo geral

* seleciono "File..." no combobox de Scene Camera
* abre uma janela com um browser para localizar e carregar o arquivo
* o arquivo é carregado com o primeiro frame do vídeo dentro do espaço para a câmera
* o botão "Calibrate" torna-se um "Play Button" também
* ao clicar em "Calibrate" inicia-se o processo de calibração e o vídeo começa a rodar



##### Abrindo vídeo no código:

* criar um método *load_video* em video_uvc.py. Esse método recebe um arquivo de vídeo e define o source da camera em questão (scene, left or right).
* criar um método *set_video_file* em camera_proc.py. Esse método é uma cópia de *set_source*, mas adaptado para um arquivo de vídeo.
* adaptar o método *run* de img_processor.py. Se o source for uma string com o nome do arquivo, então run deve carregá-lo usando opencv. 

