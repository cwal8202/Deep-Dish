 # 'image_files'라는 이름으로 전송된 파일들을 리스트로 받음
def upload_file(image_files):
    """파일 업로드 화면 및 처리"""
    uploaded_files = request.files.getlist('image_files')

    if not uploaded_files or uploaded_files[0].filename == '':
        print("파일이 선택되지 않았습니다.")
        return False

    # 1) 월별 폴더명 생성 (YYYYMM)
    month_folder = datetime.now().strftime("%Y%m")
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], month_folder), exist_ok=True)

    for file in uploaded_files:
        if file:
            # 안전한 파일 이름으로 변경
            filename = secure_filename(file.filename)
            # 파일을 지정된 폴더에 저장
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], month_folder,filename))
            print(f"'{filename}' 파일 저장 완료.")
    return True