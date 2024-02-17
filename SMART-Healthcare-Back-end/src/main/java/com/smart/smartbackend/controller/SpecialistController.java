package com.smart.smartbackend.controller;

import com.smart.smartbackend.entity.Specialist;
import com.smart.smartbackend.service.SpecialistService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.concurrent.ExecutionException;

@RestController
@RequestMapping("api/v1/specialist")
@CrossOrigin
public class SpecialistController {
    @Autowired
    SpecialistService specialistService;

    @PostMapping("/save")
    public String saveSpecialist(@RequestBody Specialist specialist) throws ExecutionException, InterruptedException {
        return  specialistService.save(specialist);
    }

    @GetMapping("/andrologist")
    public List<Specialist> getAndrologists() throws ExecutionException, InterruptedException {
        return specialistService.getAllAndrologists();
    }

    @GetMapping("/cardiologist")
    public List<Specialist> getCardiologists() throws ExecutionException, InterruptedException {
        return specialistService.getAllCardiologists();
    }

    @GetMapping("/gastroenteologist")
    public List<Specialist> getGastroenteologists() throws ExecutionException, InterruptedException {
        return specialistService.getAllGastroenteologists();
    }

    @GetMapping("/gynocologist")
    public List<Specialist> getGynocologists() throws ExecutionException, InterruptedException {
        return specialistService.getAllGynocologists();
    }

    @GetMapping("/pulmonologist")
    public List<Specialist> getPulmonologists() throws ExecutionException, InterruptedException {
        return specialistService.getAllPulmonologists();
    }

    @GetMapping("/urologist")
    public List<Specialist> getUrologists() throws ExecutionException, InterruptedException {
        return specialistService.getAllUrologists();
    }
}
