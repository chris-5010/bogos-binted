// bogo_vk.cpp - vulkan compute bogosort
// build: g++ -O2 -std=c++17 bogo_vk.cpp -o bogo_vk -lvulkan
// needs: vulkan-headers vulkan-radeon glslang
// usage: echo "seed_lo seed_hi batch offset" | ./bogo_vk --daemon [--work-items N]

#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define VK_CHECK(e) do { VkResult _r=(e); if(_r!=VK_SUCCESS){ \
    fprintf(stderr,"vk error %d at %s:%d\n",_r,__FILE__,__LINE__); exit(1);} } while(0)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// result layout: 10 x uint32 per work-item
//   [0]     score (0xffffffff = no result)
//   [1..7]  arr packed 4 bytes/uint (25 bytes total)
//   [8..9]  best_index lo/hi
#define RSTRIDE 10

static std::vector<uint32_t> compile_shader(const char *path, int bs) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "glslangValidator -V --target-env vulkan1.2 -DBLOCK_SIZE=%d "
        "-DRESULT_STRIDE=%d %s -o /tmp/bogo.spv 2>&1", bs, RSTRIDE, path);
    FILE *p = popen(cmd, "r");
    char buf[2048] = {}; fread(buf, 1, sizeof(buf)-1, p);
    if (pclose(p)) { fprintf(stderr, "glslang: %s\n", buf); exit(1); }
    std::ifstream spv("/tmp/bogo.spv", std::ios::binary | std::ios::ate);
    size_t n = spv.tellg(); spv.seekg(0);
    std::vector<uint32_t> c(n/4); spv.read((char*)c.data(), n);
    return c;
}


static void setup(Ctx &c, size_t n, int bs, const char *shader) {
    c.n=n; c.bs=bs;

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo=&ai;
    VK_CHECK(vkCreateInstance(&ici,nullptr,&c.inst));

    uint32_t nd=0; vkEnumeratePhysicalDevices(c.inst,&nd,nullptr);
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(c.inst,&nd,devs.data());
    c.pdev=devs[0];
    for (auto &d:devs) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p);
        if (p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { c.pdev=d; break; }
    }
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(c.pdev,&pp);
    fprintf(stderr,"[vk] %s\n",pp.deviceName);

    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(c.pdev,&nq,nullptr);
    std::vector<VkQueueFamilyProperties> qp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.pdev,&nq,qp.data());
    c.qfam=0;
    for (uint32_t i=0;i<nq;++i) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){c.qfam=i;break;}

    float prio=1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=c.qfam; qci.queueCount=1; qci.pQueuePriorities=&prio;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.shaderInt8=VK_TRUE; f12.storageBuffer8BitAccess=VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext=&f12; f2.features.shaderInt64=VK_TRUE;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext=&f2; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    VK_CHECK(vkCreateDevice(c.pdev,&dci,nullptr,&c.dev));
    vkGetDeviceQueue(c.dev,c.qfam,0,&c.q);

    auto spv=compile_shader(shader, bs);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize=spv.size()*4; smi.pCode=spv.data();
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(c.dev,&smi,nullptr,&sm));

    VkDescriptorSetLayoutBinding b{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount=1; dlci.pBindings=&b;
    VK_CHECK(vkCreateDescriptorSetLayout(c.dev,&dlci,nullptr,&c.dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,20};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&c.dsl;
    plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VK_CHECK(vkCreatePipelineLayout(c.dev,&plci,nullptr,&c.pl));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module=sm; cpci.stage.pName="main"; cpci.layout=c.pl;
    VK_CHECK(vkCreateComputePipelines(c.dev,VK_NULL_HANDLE,1,&cpci,nullptr,&c.pipe));
    vkDestroyShaderModule(c.dev,sm,nullptr);

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size=(VkDeviceSize)n*RSTRIDE*sizeof(uint32_t);
    bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.dev,&bci,nullptr,&c.buf));

    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev,c.buf,&mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(c.pdev,&mp);
    uint32_t mt=UINT32_MAX;
    VkMemoryPropertyFlags want=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i=0;i<mp.memoryTypeCount;++i)
        if((mr.memoryTypeBits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want){mt=i;break;}
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize=mr.size; mai.memoryTypeIndex=mt;
    VK_CHECK(vkAllocateMemory(c.dev,&mai,nullptr,&c.mem));
    VK_CHECK(vkBindBufferMemory(c.dev,c.buf,c.mem,0));
    VK_CHECK(vkMapMemory(c.dev,c.mem,0,bci.size,0,(void**)&c.mapped));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&ps;
    VK_CHECK(vkCreateDescriptorPool(c.dev,&dpci,nullptr,&c.dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=c.dpool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&c.dsl;
    VK_CHECK(vkAllocateDescriptorSets(c.dev,&dsai,&c.ds));
    VkDescriptorBufferInfo dbi{c.buf,0,bci.size};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet=c.ds; wds.dstBinding=0; wds.descriptorCount=1;
    wds.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds.pBufferInfo=&dbi;
    vkUpdateDescriptorSets(c.dev,1,&wds,0,nullptr);

    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpci2.queueFamilyIndex=c.qfam;
    VK_CHECK(vkCreateCommandPool(c.dev,&cpci2,nullptr,&c.cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=c.cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VK_CHECK(vkAllocateCommandBuffers(c.dev,&cbai,&c.cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(c.dev,&fci,nullptr,&c.fence));
}

struct Result { int score; uint8_t arr[25]; uint64_t idx; };

static Result dispatch(Ctx &c, uint32_t slo, uint32_t shi, uint32_t bpi, uint64_t off) {
    struct { uint32_t slo,shi,bpi,ilo,ihi; } pc;
    pc.slo=slo; pc.shi=shi; pc.bpi=bpi;
    pc.ilo=(uint32_t)(off&0xffffffff); pc.ihi=(uint32_t)(off>>32);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(c.cb,&bi));
    vkCmdBindPipeline(c.cb,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipe);
    vkCmdBindDescriptorSets(c.cb,VK_PIPELINE_BIND_POINT_COMPUTE,c.pl,0,1,&c.ds,0,nullptr);
    vkCmdPushConstants(c.cb,c.pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),&pc);
    vkCmdDispatch(c.cb,(uint32_t)((c.n+c.bs-1)/c.bs),1,1);
    VK_CHECK(vkEndCommandBuffer(c.cb));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount=1; si.pCommandBuffers=&c.cb;
    VK_CHECK(vkQueueSubmit(c.q,1,&si,c.fence));
    VK_CHECK(vkWaitForFences(c.dev,1,&c.fence,VK_TRUE,UINT64_MAX));
    vkResetFences(c.dev,1,&c.fence);

    Result best{-1,{},0};
    for (size_t i=0;i<c.n;++i) {
        uint32_t sc=c.mapped[i*RSTRIDE];
        if (sc==0xffffffffu) continue;
        if ((int)sc>best.score) {
            best.score=(int)sc;
            for (int w=0;w<7;++w) {
                uint32_t v=c.mapped[i*RSTRIDE+1+w];
                for (int b=0;b<4;++b) { int k=w*4+b; if(k<25) best.arr[k]=(v>>(b*8))&0xff; }
            }
            best.idx=(uint64_t)c.mapped[i*RSTRIDE+8]|((uint64_t)c.mapped[i*RSTRIDE+9]<<32);
        }
    }
    return best;
}

static void daemon(Ctx &c) {
    fprintf(stderr,"[vk] ready  n=%zu  bs=%d\n",c.n,c.bs); fflush(stderr);
    char line[256];
    while (fgets(line,sizeof(line),stdin)) {
        unsigned long long sl,sh,batch,off;
        if (sscanf(line,"%llu %llu %llu %llu",&sl,&sh,&batch,&off)!=4) continue;
        uint32_t bpi=(uint32_t)((batch+c.n-1)/c.n);
        uint64_t actual=(uint64_t)c.n*bpi;
        double t=now_s();
        Result r=dispatch(c,(uint32_t)sl,(uint32_t)sh,bpi,(uint64_t)off);
        double e=now_s()-t;
        printf("{\"best_correct\":%d,\"best_arr\":[",r.score);
        for (int i=0;i<25;i++) printf("%s%d",i?",":"",r.arr[i]);
        printf("],\"best_index\":%llu,\"total_done\":%llu,\"elapsed\":%.6f,\"rate\":%.0f}\n",
               (unsigned long long)r.idx,(unsigned long long)actual,e,e>0?actual/e:0.);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    bool dm=false; size_t n=262144; int bs=256;
    std::vector<std::string> pos;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        if (a=="--daemon") dm=true;
        else if (a=="--work-items"&&i+1<argc) n=std::stoull(argv[++i]);
        else if (a=="--block-size"&&i+1<argc) bs=std::stoi(argv[++i]);
        else if (!a.empty()&&a[0]!='-') pos.push_back(a);
    }
    n=((n+bs-1)/bs)*bs;

    // look for bogo.comp next to the binary, then current dir
    char shader_path[512] = "bogo.comp";
    if (argc > 0) {
        std::string bin = argv[0];
        auto slash = bin.rfind('/');
        if (slash != std::string::npos)
            snprintf(shader_path, sizeof(shader_path), "%s/bogo.comp",
                     bin.substr(0, slash).c_str());
    }

    Ctx c; setup(c, n, bs, shader_path);
    if (dm) { daemon(c); return 0; }

    uint32_t slo=pos.size()>0?(uint32_t)std::stoull(pos[0]):0x12345678u;
    uint32_t shi=pos.size()>1?(uint32_t)std::stoull(pos[1]):0xdeadbeef;
    uint64_t tot=pos.size()>2?std::stoull(pos[2]):2000000000ULL;
    uint32_t bpi=(uint32_t)((tot+n-1)/n);
    double t=now_s();
    Result r=dispatch(c,slo,shi,bpi,0);
    double e=now_s()-t;
    printf("{\"seed\":\"%llu\",\"best_correct\":%d,\"best_index\":%llu,\"best_arr\":[",
           (unsigned long long)(((uint64_t)shi<<32)|slo),r.score,(unsigned long long)r.idx);
    for (int i=0;i<25;i++) printf("%s%d",i?",":"",r.arr[i]);
    printf("],\"rate\":%.0f}\n",(uint64_t)n*bpi/e);
}
