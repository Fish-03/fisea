// 這個主要為了生成.EXE檔案,用於DEBUG

#include <iostream>
#include <vector>
#include <typeinfo>
   class testA : public std::enable_shared_from_this<testA>
    {
    public:

        virtual void print() = 0;
        virtual ~testA() {}
        template <typename T>
        T getdata()
        {
            const testB *b = dynamic_cast<const testB *>(this);
            if (b)
            {
                return b->data;
            }
            return 0;
        }
        
    };
    class testB : public testA, public std::enable_shared_from_this<testB>
    {
        private:
        float data = 1.5;
    protected:
        float data2 = 2.5;
    public:
        int a = 10;
        friend class testA;
        testB();
        ~testB();
        void print();
        float getdata() const;
        static std::shared_ptr<testA> create();
        
    };

    testB::testB()
    {
        std::cout << "testB" << std::endl;
    }
    testB::~testB()
    {
        std::cout << "~testB" << std::endl;
    }
    void testB::print()
    {
        std::cout << "print" << std::endl;
    }
    std::shared_ptr<testA> testB::create()
    {
        std::cout << "create" << std::endl;
        
        return std::make_shared<testB>();
    }
    float testB::getdata() const
    {
        return data;
    }
int main()
{
    // std::vector<int> shape = {2, 3, 4};
    // fisea::Tensor t = fisea::Tensor(shape, "cpu", "float");
    // t.cpu();
    // // t.cuda();
    // return 0;
    std::shared_ptr<testA> a = testB::create();
    // std::cout << a->getdata<float>() << std::endl;
    a->print();
    std::cout << a->getdata<float>() << std::endl;
    
    return 0;
}