#include <iostream>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <vector>

class Timer
{
private:
    std::chrono::_V2::system_clock::time_point now;
    std::chrono::_V2::system_clock::time_point last_time;
    std::chrono::nanoseconds dt;
    int dtMs;

    std::vector<std::string> names;
    std::vector<int> times;

public:
    Timer()    {
        last_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {}

    void reset(){
        now = std::chrono::high_resolution_clock::now();
        last_time = now;
        times.clear();
        names.clear();
    }

    // 阶段性更新计时器
    int update(std::string name){
        now = std::chrono::high_resolution_clock::now();
        dt = now - last_time;
        last_time = now;
        dtMs = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
        times.push_back(dtMs);
        names.push_back(name);
        return dtMs;
    }

    // 输出各个阶段的计时器并重置
    int print(std::string text){
        printf("[ %s ] ", text.c_str());
        int total = 0;
        for (int i=0; i<times.size(); i++)
        {
            printf(" %s: dt = %d ms, ", names[i].c_str(), times[i]);
            total += times[i];
        }
        printf(" total = %d ms\n", total);
        reset();
        return total;
    }
};