#include <stdio.h>

int cal(void){
    int taus[] = {2,25,7};
    int Lmax = 2;
    for (int l1 = 0; l1 <= Lmax; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1-l2;l<=Lmax&&l<=l1+l2;l++){
                int t_offset = 0;
                for (int templ1 = 0; templ1<l1; templ1++){
                    for (int templ2 = 0; templ2<=templ1; templ2++){
                        if (l <= templ2 + templ1 && l >= templ1- templ2){
                            t_offset += taus[templ1]*taus[templ2];  
                        }
                    }
                }
                for (int templ2 = 0; templ2<=l2; templ2++){
                    if (l <= templ2 + l1 && l >= l1- templ2){
                        t_offset += taus[l1]*taus[templ2];  
                    }
                }
                t_offset -= taus[l1]*taus[l2];

                int new_t_offset=0;
                for (int templ1=0; templ1 <= l1; templ1++){
                    for (int templ2=0; (templ2<l2 && templ1==l1) || (templ2<=templ1 && templ1<l1);templ2++){
                        if (l <= templ2+templ1 && l >= templ1-templ2){
                            new_t_offset += taus[templ1]*taus[templ2];  
                        }
                    }
                }
                printf("l1=%d,l2=%d, t_offset=%d, got %d\n", l1,l2, t_offset,new_t_offset);
            }
        }

    }

    return 0;
}

int cal2(void){

    int Lmax = 2;
    int pos = 0;
    for (int l1 = 0; l1 <= Lmax; l1++){
        for (int l2 = 0; l2 <= l1; l2++){
            for (int l = l1-l2; l<=Lmax && l <= l1+l2; l++){
                int start = 0;
                for (int templ1=0; templ1 <= l1; templ1++){
                    for (int templ2=0; (templ2<l2 && templ1==l1) || (templ2<=templ1 && templ1<l1);templ2++){
                        int low = templ1-templ2, high=(templ2+templ1 > Lmax) ? Lmax : templ2+templ1;
                        for (int templ=low; templ<=high ; templ++){
                            start += (2*templ2+1)*(2*templ+1);
                            //printf("low=%d,high=%d,%d\n",low,high, start);
                        }
                    }
                }
                for (int templ = l1-l2; templ<l; templ++){
                    start += (2*l2+1)*(templ*2+1);
                }

                printf("l1=%d l2=%d l=%d, start=%d, got %d\n",l1,l2,l,pos,start);

                pos += (2*l2+1)*(2*l+1);
            }
        }
    }

}

int main(int argc, char const *argv[])
{
    return cal();
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