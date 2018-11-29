#include <stdio.h>

int cal(void){

    int L = 30;
    int total_cnt = 0;
    for (int l1 = 0; l1 <= L; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1 - l2; l <= L && l <= l1 + l2; l++){
                total_cnt += (2 * l1 + 1) * (2 * l2 + 1);
            }
        }
    }
    printf("%d\n", total_cnt);
    return 0;
}

int cal2(void){
    int taus[] = {3,2,1};
    int Lmax = 2;
    int l = 2;
    int l1=2,l2=0;
    int t1=0,t2=0;
    int t = t1 * taus[l2] + t2;
    int t_offset = 0;
    for (int templ1 = 0; templ1<l1; templ1++){
        for (int templ2 = 0; templ2<=templ1; templ2++){
            if (l <= templ2 + templ1 && l >= templ1- templ2){
                t_offset += taus[templ1]*taus[templ2];
                printf("%d%d %d\n",templ1, templ2, t_offset);    
            }
        }
    }
    for (int templ2 = 0; templ2<=l2; templ2++){
        if (l <= templ2 + l1 && l >= l1- templ2){
            t_offset += taus[l1]*taus[templ2];
            printf("%d%d %d\n",l1, templ2, t_offset);    
        }
    }
    printf("%d\n", t_offset);
    t_offset -= taus[l1]*taus[l2];
    printf("%d\n", t_offset);
}


int main(int argc, char const *argv[])
{
    //int k = 0, k2 = 3;
    //k2 *= 2;
    //while (k*(k+1) <= k2){k++;}
    //k -= 1;
    //k2 = (k2 - k*(k+1))/2;
    //printf("%d %d\n", k, k2);
    //return 0;
    //return cal2();
    int lm = 4;
    int l1 = 0, m1 = 0;
    while (l1*l1 <= lm){l1++;}
    l1--;
    m1 = lm - l1 * l1;
    printf("%d %d\n", l1, m1);
    return 0;
    /* code */
    int taus[] = {2,1};
    int Lmax = 1;

    int l=0, t=0;
    for (l=0;l<=Lmax;l++){
        for (t = 0;t< (l == 0 ? 5: 9);t++){
            int cnt_t = 0;
            int l1 = 0, l2 = 0;
            while (l1 <= Lmax){
                l2 = 0;
                while (l2 <= l1){
                    cnt_t += (l1-l2<=l && l<=l1+l2) ? taus[l1]*taus[l2]*(2*l+1) : 0;
                    //printf("%d > %d ?\n", cnt_t, t);
                    if (cnt_t > t) break;
                    l2++;
                }
                if (cnt_t > t) break;
                l1++;
            }
            printf("l:%d, t:%d, l1:%d, l2:%d\n", l, t, l1,l2);
        }
    }
    
    return 0;
}