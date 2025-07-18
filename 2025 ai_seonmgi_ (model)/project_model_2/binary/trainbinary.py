import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# 앞서 설정한 함수 불러오기
from dataloader_binary import get_binary_dataloaders
from model_binary_resnet import get_resnet18_binary_model

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 15
learning_rate = 1e-4
save_path = "best_binary_model.pth"

def create_training_video():
    """저장된 프레임들로 영상 생성"""
    try:
        import cv2
        print("🎬 학습 과정 영상 생성 중...")
        
        # 프레임 읽기
        frame_files = sorted([f for f in os.listdir('training_video_frames') if f.endswith('.png')])
        
        if not frame_files:
            print("❌ 영상 프레임이 없습니다.")
            return
        
        # 첫 번째 프레임으로 영상 크기 설정
        first_frame = cv2.imread(os.path.join('training_video_frames', frame_files[0]))
        height, width, layers = first_frame.shape
        
        # 영상 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2  # 초당 2프레임 (느리게)
        video_writer = cv2.VideoWriter('training_animation.mp4', fourcc, fps, (width, height))
        
        # 각 프레임을 영상에 추가
        for frame_file in frame_files:
            frame_path = os.path.join('training_video_frames', frame_file)
            frame = cv2.imread(frame_path)
            
            # 각 프레임을 여러 번 추가 (더 오래 보이게)
            for _ in range(2):  # 각 프레임을 2번씩 추가
                video_writer.write(frame)
        
        video_writer.release()
        print("✅ 학습 영상 저장 완료: training_animation.mp4")
        
    except ImportError:
        print("⚠️ OpenCV가 설치되지 않아 영상을 생성할 수 없습니다.")
        print("   설치: pip install opencv-python")
    except Exception as e:
        print(f"❌ 영상 생성 실패: {e}")

def create_gif_animation():
    """GIF 애니메이션 생성 (OpenCV 없이도 가능)"""
    try:
        from PIL import Image
        print("🎞️ GIF 애니메이션 생성 중...")
        
        frame_files = sorted([f for f in os.listdir('training_video_frames') if f.endswith('.png')])
        
        if not frame_files:
            print("❌ GIF 프레임이 없습니다.")
            return
        
        # 이미지들 로드
        images = []
        for frame_file in frame_files:
            frame_path = os.path.join('training_video_frames', frame_file)
            img = Image.open(frame_path)
            images.append(img)
        
        # GIF 저장
        images[0].save(
            'training_animation.gif',
            save_all=True,
            append_images=images[1:],
            duration=800,  # 각 프레임 800ms
            loop=0
        )
        print("✅ GIF 애니메이션 저장 완료: training_animation.gif")
        
    except Exception as e:
        print(f"❌ GIF 생성 실패: {e}")

def save_plot_as_image(epoch, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies):
    """각 에포크마다 그래프를 이미지로 저장"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss 그래프
    ax1.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=3, marker='o', markersize=6)
    ax1.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=3, marker='s', markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, num_epochs)
    
    # Accuracy 그래프
    ax2.plot(epochs_list, train_accuracies, 'b-', label='Train Accuracy', linewidth=3, marker='o', markersize=6)
    ax2.plot(epochs_list, val_accuracies, 'r-', label='Val Accuracy', linewidth=3, marker='s', markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(1, num_epochs)
    
    # 현재 최고 성능 표시
    if val_accuracies:
        best_val_acc = max(val_accuracies)
        best_epoch = val_accuracies.index(best_val_acc) + 1
        ax2.axhline(y=best_val_acc, color='orange', linestyle='--', alpha=0.7)
        ax2.text(num_epochs*0.7, best_val_acc+0.02, f'Best: {best_val_acc:.4f} (Epoch {best_epoch})', 
                fontsize=10, fontweight='bold', color='orange')
    
    plt.suptitle(f'Training Progress - Epoch {epoch}/{num_epochs}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 개별 에포크 이미지 저장
    plt.savefig(f'training_plots/epoch_{epoch:02d}.png', dpi=300, bbox_inches='tight')
    
    # 영상용 프레임 저장 (더 높은 해상도)
    plt.savefig(f'training_video_frames/frame_{epoch:02d}.png', dpi=200, bbox_inches='tight')
    
    plt.close()

def update_plots(ax1, ax2, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies):
    """실시간으로 그래프 업데이트"""
    ax1.clear()
    ax2.clear()
    
    # Loss 그래프
    ax1.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax1.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 그래프
    ax2.plot(epochs_list, train_accuracies, 'b-', label='Train Acc', linewidth=2, marker='o')
    ax2.plot(epochs_list, val_accuracies, 'r-', label='Val Acc', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

# 🛡️ 메인 가드 시작 - Windows 멀티프로세싱 오류 방지
if __name__ == '__main__':
    # 저장 폴더 생성
    os.makedirs("training_plots", exist_ok=True)
    os.makedirs("training_video_frames", exist_ok=True)

    # GPU 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 데이터 불러오기
    train_loader, val_loader, test_loader = get_binary_dataloaders(
        data_dir=None,
        batch_size=batch_size,
        num_workers=4  # Windows에서도 이제 안전함
    )

    # 모델 불러오기
    model = get_resnet18_binary_model(pretrained=True).to(device)

    # 손실 함수 & 옵티마이저 & 스케줄러 설정
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 📊 학습 기록을 위한 리스트
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_list = []

    # 🎨 그래프 스타일 설정
    plt.style.use('default')  # seaborn 대신 기본 스타일 사용
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # 🎨 실시간 그래프 설정
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 학습 루프
    best_val_acc = 0.0
    print("\n🚀 학습 시작! (실시간 그래프 + 이미지/영상 저장)")

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
                
        # Training Phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        avg_train_loss = train_loss / total
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        avg_val_loss = val_loss / total
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 📊 기록 저장
        epochs_list.append(epoch + 1)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 🎨 실시간 그래프 업데이트
        update_plots(ax1, ax2, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies)
        
        # 📸 에포크별 그래프 이미지 저장
        save_plot_as_image(epoch + 1, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies)
        scheduler.step()

        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to: {save_path}")

    print(f"\n🎉 학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")

    # 📊 최종 그래프 저장
    plt.ioff()
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=3, marker='o', markersize=8)
    plt.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=3, marker='s', markersize=8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Final Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_accuracies, 'b-', label='Train Accuracy', linewidth=3, marker='o', markersize=8)
    plt.plot(epochs_list, val_accuracies, 'r-', label='Val Accuracy', linewidth=3, marker='s', markersize=8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Final Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.suptitle(f'Training Results - Best Val Acc: {best_val_acc:.4f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('final_training_results.png', dpi=300, bbox_inches='tight')
    print("📊 최종 학습 그래프 저장: final_training_results.png")
    plt.show()

    # 🎬 영상 및 GIF 생성
    create_training_video()
    create_gif_animation()

    # 📈 학습 기록을 CSV로 저장
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    })
    history_df.to_csv('training_history.csv', index=False)
    print("📁 학습 기록 저장: training_history.csv")

    print("\n📁 생성된 파일들:")
    print("   🖼️ training_plots/ - 에포크별 그래프 이미지들")
    print("   🎬 training_animation.mp4 - 학습 과정 영상")
    print("   🎞️ training_animation.gif - 학습 과정 GIF")
    print("   📊 final_training_results.png - 최종 결과 그래프")
    print("   📋 training_history.csv - 수치 데이터")
    
# training_plots/epoch_01.png, epoch_02.png, ... (각 에포크별 그래프)
# final_training_results.png (최종 결과 그래프)
# training_animation.mp4 (학습 과정 영상)
# training_animation.gif (학습 과정 GIF 애니메이션)
# training_history.csv (수치 데이터)