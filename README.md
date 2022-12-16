# 차선인식 

<aside>
팀명 : unbreako

팀원 : 송원석, 한지호

</aside>

# 프로젝트 기획

## 프로젝트 목표
![목표](https://user-images.githubusercontent.com/62413303/208063032-e54a02da-de23-4323-a8ef-295289dfe1ef.png)

- 영상의 밝기가 변화하는 상황에서 차선을 지속적으로 추적할 수 있도록 다양한 영상처리 기법 활용
- 자이카의 전면부에 부착된 라이다에 차선 인식이 방해받지 않도록 조정
- 주변 지형 지물물을 차선으로 인식하지 않도록 조정
- 차선이 끊긴 후 다시 시작되는 구간에서 차선 다시 포착

**해결할 문제 4가지**

- [x]  영상 밝기의 변화
- [x]  Offset에 가려진 LiDAR
- [x]  지형 지물 필터링
- [x]  차선의 끊김

## 프로젝트에 사용한 알고리즘

- Histogram Streching
- Image maksing
- Average filter
- Hough transform

## 프로젝트 전략
![전략](https://user-images.githubusercontent.com/62413303/208063059-b5a4ebc8-0c37-43c1-8da5-8d25e3bb5f48.png)
 저희 팀은 차선 인식을 위한 주요 알고리즘으로 Hough transform을 채택했습니다. 그 이유는 팀원들이 명도 기반 차선 인식이나 슬라이딩 도어 기법보다 허프 트랜스폼 기법에 대한 이해도가 높아서 프로젝트 기간 동안 최대한의 퍼포먼스를 낼 수 있을 것이라고 판단했기 때문입니다.
  저희 팀의 전략은 크게 네 부분으로 나눌 수 있습니다.

   1. 허프 함수가 차선을 잘 추출해낼 수 있도록 다양한 영상 전처리를 진행합니다.
   2. 허프 변환을 진행하여 전처리를 진행한 영상에서 선분들을 추출합니다.
   3. 평균값 필터를 사용하여 추출된 선분들을 한번 더 필터링합니다.
   4. 필터링을 거친 선분들을 사용하여 차선의 위치를 구합니다.

# 프로젝트 결과

## 최종 결과
![제목 없는 동영상 - Clipchamp로 제작](https://user-images.githubusercontent.com/62413303/208063504-e9f27185-966c-4083-b30d-ae253acc28ee.gif)
- 최종 결과물은 위의 영상으로 확인할 수 있습니다. 보시다시피 꽤 정확하게 인식을 하는 것을 확인할 수 있습니다.
- 정확도를 구하기 위해 실제로 수기로 작성한 데이터 파일과 비교해보았습니다.
- 양쪽 차선이 모두 차선 범위 안으로 들어간 경우만 +1점으로 채점했을 때 최종 정확도 **80%**을 얻을 수 있었습니다.

## 협업 과정

- 구글 Meet을 사용하여 대화를 하면서 아이디어를 도출하고 코드를 작성하였습니다.

## 이슈사항

<aside>
💡 저희는 프로젝트를 진행하면서 총 4가지의 이슈상황들이 있었습니다.

</aside>

- 지형 지물에 의해 외부값으로 차선이 튀는 상황

    기둥이나 바닥, 책상 등 기물의 가장자리가 캐니 에지 → 허프 변환의 과정에서 직선으로 인식되어 차선 인식에 방해되는 상황이 발생하여 평균값 필터를 통해 이를 해결하였습니다.
     필터의 평균값은 이전 10픽셀의 왼쪽, 오른쪽 차선 x좌표를 기억하여 최근 좌표일수록 가중치를 주고, 평균을 내어 구하였습니다. HoughLinesP 함수의 반환선분 중 x좌표의 평균값이 필터의 평균값과 일정 이상 차이가 날 경우 필터링하고 남은 값들로 차선을 인식했습니다. 차선이 끊길 경우 새로운 차선이 발견되었을 때 이전의 평균값이 방해되지 않도록 이전 10픽셀의 값들과 필터링 값을 모두 초기화 하였습니다.
     알고리즘을 도입한 이후 이전보다 지형지물을 차선으로 인식하는 경우가 훨씬 적고, 정확도가 향상되는 효과를 얻었습니다.

- 라이다가 차선이랑 겹치는 순간 인식
    
    차선의 좌표를 구해야 하는 영상의 y=400범위에서 라이다의 끝부분이 걸쳐서 자이카의 각도에 따라 라이다가 차선으로 인식되는 경우가 발생했습니다. 저희 팀은 라이다가 차지하는 픽셀 값들을 캐니 에지에 의해 이진화된 검정색과 같은 색으로 Masking 하는 해결 방법을 사용했습니다. 아래의 사진이 실제로 저희가 사용한 Masking 이미지입니다.
    
    영상의 부분을 추출해서 그 안의 선분을 추출하는 허프 알고리즘의 특성 상 라이다 외에도 좌 우의 돌출도 차선 인식에 방해가 되는 경우가 발생했습니다. 직선을 인식하는 영상의 부분을 조절하고, HoughLinesP의 인수 값을 조절하여 돌출과 겹치지 않는 구간의 선분들을 더 추출해낼수 있도록 조정하였습니다.
    결과 영상 상에서 라이다나 돌출의 선분을 차선으로 인식하거나, 차선과 라이다의 경계선이 겹쳐서 허프 함수가 선분을 추출해내지 못하는 경우가 현저히 감소했습니다.
    
- 곡선인식
    
     프로젝트 초기에 라이다를 인식하는 것을 피하기 위해 영상의 중심점에서 일정 픽셀 이상 떨어진 선분들만 이용하여 차선을 추출했기 때문에 곡선 구간에서 차선이 영상의 중심점과 가까워지면 인식하지 못하는 이슈가 있었습니다.
     라이다을 인식하는 이슈를 해결하는 과정에서 라이다에 마스크를 씌우면서 중심점 부근의 제한을 해제할 수 있게 되었고, 이를 통해 자연스럽게 곡선 구간의 인식 문제가 많이 개선되었습니다.
    
- 수직선분 인식
    
    저희 팀이 사용한 알고리즘은 차선을 인식하는 과정에서 선분의 기울기를 사용하기 때문에, 기울기가 0이거나 과도하게 큰 값이 나올 경우 중간에 필터링을 거쳤습니다. 그 결과 수평인 선분과 수직에 가까운 선분들은 추출해내지 못하는 문제상황이 발생하였습니다.
    이를 해결하기 위해 우선 영상에서 프로그램을 실행시켰을 때 HoughLinesP 함수가 반환하는 선분의 x1과 x2값이 일치하는 경우가 없고, 에러가 발생하지 않는 것을 확인한 후, 기울기 값의 상한을 제거했습니다. 그리고 차선이 너무 얇아서 수직선을 검출하지 못하는 부분을 해결하기 위해 dilate 함수를 사용하여 추출하는 선분의 양을 늘리고, 허프 함수의 인수를 조절하여 수직선도 잘 인식하도록 개선하였습니다.
    
